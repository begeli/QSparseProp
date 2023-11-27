from modules.linear import QuantizedSparseLinear
from modules.conv2d import QuantizedSparseConv2d
from modules.sparseprop.linear import SparseLinear
from modules.sparseprop.conv2d import SparseConv2d
from utils.network_utils import swap_module, sparsity

import os
from collections import OrderedDict
from copy import deepcopy
from time import time
import torch
from tqdm.auto import tqdm


class Finetuner:
    def __init__(self, model, optimizer, schedular, loss_fn, log_freq, save_freq, logger):
        self._model = WrappedModel(model) # TODO: Comment here
        #self._model = model
        self._optimizer = optimizer
        self._schedular = schedular
        self._loss_fn = loss_fn
        self._log_freq = log_freq
        self._save_freq = save_freq
        self._logger = logger

    def _step(self, inputs, targets, phase, timings=None, profile=False):
        assert phase in ['train', 'test']
        train = phase == 'train'

        if timings is None:
            timings = OrderedDict()

        if train:
            self._optimizer.zero_grad()

        with Timer(timings, 'end_to_end_forward'):
            if profile:
                with torch.autograd.profiler.profile(profile_memory=True) as prof:
                    with torch.profiler.record_function("Forward Propagation"):
                        outputs, _ = self._model(inputs)

                print(
                    "================================================ Forward Profile ================================================")
                print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=20))
            else:
                outputs, _ = self._model(inputs)

        loss = self._loss_fn(outputs, targets)

        with Timer(timings, 'end_to_end_backward'):
            if train:
                if profile:
                    with torch.autograd.profiler.profile(profile_memory=True) as prof:
                        with torch.profiler.record_function("Backpropagation"):
                            loss.backward()
                    print(
                        "================================================ Backward Profile ================================================")
                    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=20))
                else:
                    loss.backward()

        if train:
            self._optimizer.step()
            if self._schedular is not None:
                self._schedular.step()

            # Requantize updated weights
            with torch.no_grad():
                for name, module in self._model.named_modules():
                    if any([isinstance(module, c) for c in [QuantizedSparseLinear, QuantizedSparseConv2d]]):
                        module.quantize_weights()

        pred = torch.argmax(outputs, dim=-1)
        acc = torch.mean((pred == targets).float())

        return loss.item(), acc.item(), timings

    def _log_timings(self, step, full_timings, modules_forward_time, modules_backward_time):
        l = f'Timings: '

        for key, value in full_timings.items():
            l += f'avg_{key}={value / (step + 1):.4f}, '
        l += f'avg_module_forward_sum={modules_forward_time:.4f}, '
        l += f'avg_module_backward_sum={modules_backward_time:.4f}'

        self._logger.log(l)


    def _run_epoch(self, loader, epoch, phase):
        #def _trace_handle(p):
        #    print(
        #        "================================================ Epoch Profile ================================================")
        #    print(p.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

        assert phase in ['train', 'test']
        train = phase == 'train'

        initial_training = self._model.training
        self._model.train(train)

        running_loss = 0.
        running_acc = 0.

        full_timings = None
        with torch.set_grad_enabled(train):
            for step, (inputs, targets) in tqdm(enumerate(loader), total=len(loader)):
                targets = torch.squeeze(targets)
                timings = OrderedDict()

                with Timer(timings, 'end_to_end_minibatch'):
                    loss, acc, _ = self._step(inputs, targets, phase, timings=timings, profile=False)

                running_loss += loss
                running_acc += acc

                if full_timings is None:
                    full_timings = timings
                else:
                    for name in timings:
                        full_timings[name] += timings[name]

                if train and (step + 1) % self._log_freq == 0:
                    avg_loss = running_loss / (step + 1)
                    avg_acc = running_acc / (step + 1)
                    self._logger.log(
                        f'[{"Train" if train else "Val"}] Epoch {epoch + 1}, Step {step + 1}: loss={avg_loss:.4f}, acc={avg_acc:.4f}')
                    self._log_timings(step, full_timings, *self._model.get_timings_sum()) # TODO: Comment here

                #break # TODO: Delete later

        avg_loss = running_loss / (step + 1)
        avg_acc = running_acc / (step + 1)
        self._logger.log(f'[{"Train" if train else "Val"}] Epoch {epoch + 1}: loss={avg_loss:.4f}, acc={avg_acc:.4f}')
        self._log_timings(step, full_timings, *self._model.get_timings_sum(reset_afterwards=True)) # TODO: Comment here

        self._model.train(initial_training)

    def finetune(self, train_loader, test_loader, epochs):
        timings = OrderedDict()

        with Timer(timings, 'end_to_end_finetuning'):
            for epoch in range(epochs):
                with Timer(timings, f'epoch_train'):
                    self._run_epoch(train_loader, epoch, phase='train')

                self._logger.log(f"Epoch {epoch + 1} training took {timings['epoch_train']:.4f}.")

                with Timer(timings, f'epoch_val'):
                    self._run_epoch(test_loader, epoch, phase='test')

                self._logger.log(f"Epoch {epoch + 1} validation took {timings['epoch_val']:.4f}.")

                # Checkpoint -- TODO: Probably need to handle optimizer state as well
                #if (epoch + 1) % self._save_freq == 0:
                #    torch.save(
                #        self._model.unwrap().state_dict(),
                #        os.path.join(self._logger._outdir, f'checkpoint-{epoch + 1}.pt')
                #    )

        self._logger.log(f"The full finetuning took {timings['end_to_end_finetuning']:.4f}.")
        #torch.save(self._model.unwrap().state_dict(), os.path.join(self._logger._outdir, 'final_checkpoint.pt'))


# this class can be used as a context manager to time events
class Timer(object):
    def __init__(self, target_dict, event_name):
        self._target_dict = target_dict
        self._event_name = event_name

    def __enter__(self):
        self._start = time()

    def __exit__(self, type, value, traceback):
        end = time()
        self._target_dict[self._event_name] = end - self._start


class TimingHook:
    def __init__(self, tag=None, verbose=False):
        self.clear()

    def __call__(self, module, inp, out):
        self.times.append(time())

    def clear(self):
        self.times = []


class Logger:
    def __init__(self, outdir=None):
        self._outdir = outdir

    def log(self, l):
        if self._outdir is not None:
            with open(os.path.join(self._outdir, 'log.txt'), 'a') as f:
                f.write(l)
                f.write('\n')
        print(l)


class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self._model = model
        self.forward = self._model.forward

        bwd_hooks, fwd_hooks, bwd_handles, fwd_handles = self._prepare_for_timing()
        self._bwd_hooks = bwd_hooks
        self._fwd_hooks = fwd_hooks
        self._handles = bwd_handles + fwd_handles

    def _clear_hooks(self):
        for hook_pair in self._bwd_hooks + self._fwd_hooks:
            _, hook1, hook2 = hook_pair
            hook1.clear()
            hook2.clear()

    def _prepare_for_timing(self):
        # we swap each module with a torch.nn.Sequential consisting of [identity_before, module, identity_after]
        # we employ hooks for per-layer timing:
        # for forward: (time right after module's forward) - (time right after identity_before's forward)
        # for backward: (time right after module's backward) - (time right after identity_after's backward)

        backward_hooks, forward_hooks = [], []
        backward_handles, forward_handles = [], []

        for name, module in self._model.named_modules():
            if isinstance(module, torch.nn.ReLU) or isinstance(module, torch.nn.ReLU6):
                module.inplace = False

            if isinstance(module, torch.nn.Conv2d) or isinstance(module, QuantizedSparseConv2d) \
                    or isinstance(module, torch.nn.Linear) or isinstance(module, QuantizedSparseLinear)\
                    or isinstance(module, SparseConv2d) or isinstance(module, SparseLinear):
                identity_after = torch.nn.Identity()
                identity_before = torch.nn.Identity()

                identity_backward_hook = TimingHook()
                module_backward_hook = TimingHook()
                backward_hooks.append((name, module_backward_hook, identity_backward_hook))
                backward_handles += [
                    module.register_full_backward_hook(module_backward_hook),
                    identity_after.register_full_backward_hook(identity_backward_hook)
                ]

                identity_forward_hook = TimingHook()
                module_forward_hook = TimingHook()
                forward_hooks.append((name, module_forward_hook, identity_forward_hook))
                forward_handles += [
                    module.register_forward_hook(module_forward_hook),
                    identity_before.register_forward_hook(identity_forward_hook)
                ]

                swap_module(self._model, name, torch.nn.Sequential(identity_before, module, identity_after))

        return backward_hooks, forward_hooks, backward_handles, forward_handles

    def unwrap(self, inplace=False):
        if inplace:
            model = self._model
            for handle in self._handles:
                handle.remove()
        else:
            model = deepcopy(self._model)

        replaced_module_names = [h[0] for h in self._fwd_hooks]
        for name, module in model.named_modules():
            if name in replaced_module_names:
                # replace the torch.nn.Sequential with the original module
                swap_module(model, name, module[1])

        return model

    def _safe_mean(self, arr):
        if len(arr) == 0:
            return 0.
        return sum(arr) / len(arr)

    def get_per_layer_timings(self, reset_afterwards=False):
        per_layer_forward_time = OrderedDict()
        per_layer_backward_time = OrderedDict()

        for name, conv_hook, identity_hook in self._fwd_hooks:
            per_layer_forward_time[name] = self._safe_mean(conv_hook.times) - self._safe_mean(identity_hook.times)

        for name, conv_hook, identity_hook in self._bwd_hooks:
            per_layer_backward_time[name] = self._safe_mean(conv_hook.times) - self._safe_mean(identity_hook.times)

        if reset_afterwards:
            self._clear_hooks()

        return per_layer_forward_time, per_layer_backward_time

    def get_timings_sum(self, reset_afterwards=False):
        per_layer_forward_time, per_layer_backward_time = self.get_per_layer_timings(
            reset_afterwards=reset_afterwards
        )

        return sum(per_layer_forward_time.values()), sum(per_layer_backward_time.values())
