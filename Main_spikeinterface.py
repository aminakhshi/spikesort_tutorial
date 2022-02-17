#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 00:37:08 2022

@author: amin
"""

import numpy as np
import matplotlib.pylab as plt
plt.rcParams["figure.figsize"] = (20, 12)

from scipy import signal
import os
from pathlib import Path
# import spikeinterface as si  # import core only
# import spikeinterface.extractors as se
# import spikeinterface.toolkit as st
# import spikeinterface.sorters as ss
# import spikeinterface.comparison as sc
# import spikeinterface.widgets as sw
import spikeinterface.full as si
from probeinterface.plotting import plot_probe


from spikeinterface.sortingcomponents import detect_peaks
from spikeinterface.sortingcomponents import localize_peaks

local_path = Path('/home/amin/tabarlab/data/')
preprocess_folder = local_path / 'dataset_preprocessed'
waveform_folder = local_path / 'dataset_waveforms'
recording = si.read_spikeglx(local_path)

# global kwargs for parallel computing
job_kwargs = dict(
    n_jobs=16,
    chunk_memory='10M',
    progress_bar=True,
)


# =============================================================================
# Generating a manual probe for recording
# =============================================================================
from scipy.io import loadmat
from probeinterface import Probe
from probeinterface.plotting import plot_probe

def probe_geometry(path):
    result = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.mat'):
                result.append(os.path.join(root, file))
    return result
probe_file = probe_geometry(local_path)
probe = loadmat(probe_file[0])
x, y = probe['xcoords'].ravel(), probe['ycoords'].ravel()
channel_indices = probe['chanMap0ind'].ravel()
positions = np.vstack((x, y)).T
probe = Probe(ndim=2, si_units="um")
probe.set_contacts(positions=positions, shapes='circle', shape_params={'radius': 5})
probe.set_device_channel_indices(channel_indices)
print(probe)
plot_probe(probe, with_channel_index=True)

# probe_3d = probe.to_3d(plane='xz')
# plot_probe(probe_3d)
# plt.show()
# =============================================================================
# print raw recording details
# =============================================================================
channel_ids = recording.get_channel_ids()
fs = recording.get_sampling_frequency()
num_chan = recording.get_num_channels()
num_seg = recording.get_num_segments()

print('Channel ids:', channel_ids)
print('Sampling frequency:', fs)
print('Number of channels:', num_chan)
print('Number of segments:', num_seg)

recording = recording.set_probe(probe)
plot_probe(probe)
# =============================================================================
# plotting first 5 seconds of the top 5 channels in raw recording
# =============================================================================
t_start = 0
t_end = int(5*fs)
si.plot_timeseries(recording, time_range=(0, 5), channel_ids=recording.channel_ids[0:5])

# fig, ax = plt.subplots(nrows=5, sharex='row', figsize = (10,6))
# for ind in range(5):
#     ax[ind].plot(np.arange(t_start, t_end, 1/fs), recording.get_traces(segment_index=0)[t_start:t_end, ind])
#     ax[ind].set_ylabel(channel_ids[ind], fontsize = 15)
# =============================================================================
# Preprocessing including bandpas filter, notch filter, median removal, etc.    
# =============================================================================
# bandpass between [300, 3000] Hz
recording_bp = si.preprocessing.bandpass_filter(recording, freq_min=300, freq_max=3000)
print(recording_bp)

# notch filter to remove 2000 Hz frequency components 
recording_notch = si.preprocessing.notch_filter(recording, freq=2000, q=30)
print(recording_notch)

# plotting the power spectrums for comparison
f_raw, p_raw = signal.welch(recording.get_traces(segment_index=0)[:, 0], fs=fs)
f_bp, p_bp = signal.welch(recording_bp.get_traces(segment_index=0)[:, 0], fs=fs)
f_notch, p_notch = signal.welch(recording_notch.get_traces(segment_index=0)[:, 0], fs=fs)

fig, ax = plt.subplots(figsize = (6, 4 ))
ax.semilogy(f_raw, p_raw, f_bp, p_bp, f_notch, p_notch)
ax.legend(['raw', 'bpass', 'notch'], fontsize = 12)
ax.set_xlabel('Frequency (Hz)', fontsize = 15)
ax.set_ylabel('PSD (V**2/Hz)', fontsize = 15)
plt.tight_layout()

# base noise removal filter; ('common median', 'average', 'single median or average', etc.)
recording_car = si.common_reference(recording_bp, reference='global', operator='average')
recording_cmr = si.common_reference(recording_bp, reference='global', operator='median')
recording_single = si.common_reference(recording_bp, reference='single', ref_channel_ids=['imec0.ap#AP0'])
# recording_single_groups = st.common_reference(recording_bp, reference='single',
#                                               groups=[list(channel_ids[0:2]), list(channel_ids[2:4]),
#                                               ref_channel_ids=list(channel_ids[0:2])

# plotting the output of different filter removals for comparison
fig, ax = plt.subplots(figsize = (10, 6))
ax.plot(recording.get_traces(segment_index=0)[:, 0])
ax.plot(recording_bp.get_traces(segment_index=0)[:, 0])
ax.plot(recording_car.get_traces(segment_index=0)[:, 0])
ax.plot(recording_cmr.get_traces(segment_index=0)[:, 0])
ax.plot(recording_single.get_traces(segment_index=0)[:, 0])
ax.legend(['raw', 'bpass', 'CAR', 'CMR', 'SMR'], fontsize = 12)
ax.set_xlabel('time', fontsize = 15)
ax.set_ylabel('V (mV)', fontsize = 15)

recording_cmr.save(folder=preprocess_folder, **job_kwargs)
print(recording_cmr)
# load back
recording_cmr = si.load_extractor(preprocess_folder)
recording_cmr
# si.plot_timeseries(recording_cmr, time_range=(100, 110), channel_ids=recording_cmr.channel_ids[0:10])
si.plot_timeseries(recording_cmr, time_range=(0, 5), channel_ids=recording_cmr.channel_ids[0:5])

#  estimate noise
noise_levels = si.get_noise_levels(recording_cmr, return_scaled=False)
fig, ax = plt.subplots(figsize=(8,6))
ax.hist(noise_levels, bins=np.arange(0,10, 1))
ax.set_title('noise across channel')
# =============================================================================
# Spike sorting preparation
# =============================================================================
# Get the list of all available sorters and installed sorters
si.available_sorters()
si.installed_sorters()

# Changing the default parameters of a spike sorter
default_TDC_params = si.TridesclousSorter.default_params()
print(default_TDC_params)
# tridesclous spike sorting
default_TDC_params['freq_min'] = 300.
default_TDC_params['freq_max'] = 3000.

# parameters set by params dictionary
sorting_TDC_5 = si.run_tridesclous(recording=recording_cmr, output_folder='sort_TDC_5',
                                   **default_TDC_params, )

# intsall a new spike sorter
!git clone https://github.com/MouseLand/Kilosort
kilosort_path = './Kilosort3'
ss.Kilosort3Sorter.set_kilosort3_path(kilosort_path)

ks_sorter = si.run_kilosort3(recording, output_folder='sort_ks')
print('Units found with Kilosort', ks_sorter.get_unit_ids())
 
# =============================================================================
# Applying PCA on filtered data   
## TODO: adding a parser to parse manual sorting files
# TO BE UPDATED
# =============================================================================

we = si.extract_waveforms(recording, sorting, waveform_folder,
                          load_if_exists=True,
                          ms_before=1, ms_after=2., max_spikes_per_unit=500,
                          n_jobs=1, chunk_size=30000)
print(we)

# =============================================================================
# Applying PCA on filtered data    
# =============================================================================
colors = ['Olive', 'Teal', 'Fuchsia']

pc = si.compute_principal_components(we, load_if_exists=True,
                                     n_components=3, mode='by_channel_local')
print(pc)

fig, ax = plt.subplots()
for i, unit_id in enumerate(sorting.unit_ids[:3]):
    comp = pc.get_components(unit_id)
    print(comp.shape)
    color = colors[i]
    ax.scatter(comp[:, 0, 8], comp[:, 1, 8], color=color)
    
    
all_labels, all_components = pc.get_all_components()
print(all_labels[:40])
print(all_labels.shape)
print(all_components.shape)

cmap = plt.get_cmap('Dark2', len(sorting.unit_ids))

fig, ax = plt.subplots()
for i, unit_id in enumerate(sorting.unit_ids):
    mask = all_labels == unit_id
    comp = all_components[mask, :, :]
    ax.scatter(comp[:, 0, 8], comp[:, 1, 8], color=cmap(i))

# =============================================================================
# Getting spike waveforms    
# =============================================================================

fig, ax = plt.subplots()
for i, unit_id in enumerate(sorting.unit_ids[:3]):
    wf = we.get_waveforms(unit_id)
    color = colors[i]
    ax.plot(wf[:, :, 8].T, color=color, lw=0.3)
    
    
fig, ax = plt.subplots()
for i, unit_id in enumerate(sorting.unit_ids[:3]):
    template = we.get_template(unit_id)
    color = colors[i]
    ax.plot(template[:, 8].T, color=color, lw=3)
    
    
extremum_channels_ids = si.get_template_extremum_channel(we, peak_sign='neg')
print(extremum_channels_ids)

