import random
import copy

import numpy as np
import awkward as ak
import uproot
import torch
from numba import njit

def superbatch_iterator(files, keys, superbatch_size=1024*100):
    """
    Reduces the overhead caused by reading data in small batches.
    Now the batch size and data load size are independent.
    WARNING: data load size should be a multiple of batch size.

    Parameters
    ----------
    files: file paths as in uproot.iterator
    keys: TBranch names in list
    superbatch_size: batch_size * multiplet 
    """
    arrays = []
    num_entries = 0
    iterator = uproot.iterate(files, keys, step_size=superbatch_size, library="ak", 
                             #  decompression_executor=ThreadPoolExecutor(max_workers=2) # -- parallel xz/zstd)
                              ) 

    for it in iterator:
        it_length = it.type.length

        if num_entries + it_length < superbatch_size:
            # Batch is missing some entries, get more entries
            arrays.append(it)
            num_entries += it_length
        else:
            # If batch is ready or more than ready, handle it
            remaining = superbatch_size - num_entries

            if remaining > 0:
                arrays.append(it[:remaining])
                yield ak.concatenate(arrays)  # next iteration starts directly after this line.
                arrays = [it[remaining:]]
                num_entries = it_length - remaining
            else:
                yield ak.concatenate(arrays)  # next iteration starts directly after this line.
                arrays = []
                num_entries = 0
                # arrays = [it]
                # num_entries = it_length

    if arrays:
        yield ak.concatenate(arrays)





class ModifiedUprootIterator(torch.utils.data.IterableDataset):
    def __init__(self, files, branches, shuffle=False, nWorkers=1, step_size=100):
        print('Initialize iterable dataset (Main Process)')
        self.files = files
        self.branches = branches
        # Flatten branches
        self.branchList = [b for key, value in branches.items() if value is not None for b in value]
        
        self.step_size = step_size
        self.shuffle = shuffle
        
        # We don't create iterators or split files here anymore.
        # We just store the raw configuration.
        self.sig_iterator = None
        self.bkg_iterator = None
        
        self._global_event_counter = 0

    def _get_worker_slice(self, file_list):
        """Helper to slice files for the current worker."""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single process mode: return all files
            return file_list
        else:
            # Multi-process mode:
            # Worker 0 gets index 0, 4, 8...
            # Worker 1 gets index 1, 5, 9...
            per_worker = int(np.ceil(len(file_list) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            
            # Simple slicing (stride) is the easiest way to shard
            return file_list[worker_id::worker_info.num_workers]

    def __iter__(self):
        print('__iter__ is called (inside a Worker).')
        
        # 1. Identify which files belong to THIS worker
        my_sig_files = self._get_worker_slice(self.files['sig'])
        
        if self.files.get('bkg'):
            my_bkg_files = self._get_worker_slice(self.files['bkg'])
        else:
            my_bkg_files = None

        # 2. Shuffle locally if requested
        if self.shuffle:
            random.shuffle(my_sig_files)
            if my_bkg_files:
                random.shuffle(my_bkg_files)

        # 3. Create the iterators JUST for this worker
        # Note: We create a fresh iterator every epoch (every time __iter__ is called)
        self.sig_iterator = superbatch_iterator(my_sig_files, self.branchList, superbatch_size=self.step_size)
        
        if my_bkg_files:
            self.bkg_iterator = superbatch_iterator(my_bkg_files, self.branchList, superbatch_size=self.step_size)
        else:
            self.bkg_iterator = None
            
        # 4. Handle Step Size warmup
        if self.step_size < 200: 
            self.step_size += 25
            print(f'Worker {torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 0}: step_size -> {self.step_size}')

        return self

    def __next__(self):
        # We no longer need to look up worker_id lists.
        # We just use the local iterators created in __iter__

        if self.bkg_iterator:
            xBkg = next(self.bkg_iterator)
            
            # Handle Signal Exhaustion (Signal is often smaller than Background)
            try:
                xSig = next(self.sig_iterator)
            except StopIteration:
                # If signal runs out, reload/restart it for this worker
                worker_info = torch.utils.data.get_worker_info()
                wid = worker_info.id if worker_info else 0
                print(f'Worker {wid}: Signal exhausted. Looping signal.')
                
                # Re-init signal iterator
                my_sig_files = self._get_worker_slice(self.files['sig'])
                if self.shuffle: random.shuffle(my_sig_files)
                self.sig_iterator = superbatch_iterator(my_sig_files, self.branchList, superbatch_size=self.step_size)
                xSig = next(self.sig_iterator)
            
            self.x = ak.concatenate([xBkg, xSig])
        else:
            self.x = next(self.sig_iterator)

        if self.shuffle:
            self.x = self._shuffle_akArr(self.x)
            
        self._add_four_vector_branches()
        self._add_event_index_branch()
        
        return self.x

    def _add_event_index_branch(self):
        """
        Create unique event IDs.
        To ensure uniqueness across workers, we can combine WorkerID and Counter.
        """
        n_events = len(self.x)
        
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        
        # Create a unique ID: WorkerID * 100 Billion + Counter
        # This ensures Worker 0 has 0...100, Worker 1 has 100000000000...
        base_id = (worker_id * 100_000_000_000) + self._global_event_counter
        
        evt_idx = ak.Array(np.arange(base_id, base_id + n_events, dtype=np.int64))
        self.x["event_idx"] = evt_idx
        
        self._global_event_counter += n_events

    def _shuffle_akArr(self, x):
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]
        
    def _add_four_vector_branches(self):
        # (Your existing logic here remains unchanged)
        if all(x in self.branchList for x in ['SDVTrack_pt', 'SDVTrack_eta', 'SDVTrack_phi']) and \
           any(x not in self.branchList for x in ['SDVTrack_E', 'SDVTrack_px', 'SDVTrack_py', 'SDVTrack_pz']):
            
            self.branches['tk'].extend(['SDVTrack_E', 'SDVTrack_px', 'SDVTrack_py', 'SDVTrack_pz'])
            E, px, py, pz = ptetaphim_to_epxpypz(self.x['SDVTrack_pt'], self.x['SDVTrack_eta'], self.x['SDVTrack_phi'])
            self.x['SDVTrack_E'] = E
            self.x['SDVTrack_px'] = px
            self.x['SDVTrack_py'] = py
            self.x['SDVTrack_pz'] = pz

def ptetaphim_to_epxpypz(pt, eta, phi, m=0.13957):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = np.sqrt(px*px + py*py + pz*pz + m*m)
    return (E, px, py, pz)




def stable_iterator(files, keys, superbatch_size=100, drop_last=False):
    """
    Retrieves same number of entries in each iteration for stable training.

    WARNING: Set drop_last=False in prediction mode,
             otherwise last batch will be dropped!!

    Parameters
    ----------
    files: file paths as in uproot.iterator
    keys: TBranch names in list
    superbatch_size: batch_size * multiplet 
    """
    arrays = []
    num_entries = 0
    iterator = uproot.iterate(files, keys, step_size=superbatch_size, library="ak",
                              ) 

    for it in iterator:
        it_length = it.type.length

        if num_entries + it_length < superbatch_size:
            # Batch is missing some entries, get more entries
            arrays.append(it)
            num_entries += it_length
        else:
            # If batch is ready or more than ready, handle it
            remaining = superbatch_size - num_entries

            if remaining > 0:
                arrays.append(it[:remaining])
                yield ak.concatenate(arrays)  # next iteration starts directly after this line.
                arrays = [it[remaining:]]
                num_entries = it_length - remaining
            elif remaining == 0:
                yield ak.concatenate(arrays)  # next iteration starts directly after this line.
                arrays = []
                num_entries = 0
            else:
                raise RuntimeError("This basically should not happen.")


    # Sometimes last batch has only 1 element which is not enough for BatchNorm layers.
    # They fail the training quite unexpectedly, after many epochs.
    # Handle the last batch
    if arrays and not drop_last:
        yield ak.concatenate(arrays)
    else:
        pass # Last batch is dropped


def _prewarm(loader, n_batches=1):
    it = iter(loader)
    for _ in range(n_batches):
        try:
            next(it)
        except StopIteration:
            break
    del it