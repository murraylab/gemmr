#!/usr/bin/env octave

# Script to generate PALM permutation indices for HCP data
# ========================================================
#
# cf: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/PALM/ExchangeabilityBlocks#EBs_for_data_of_the_Human_Connectome_Project
#
# Requires:
# - hcp2blocks.m
# - PALM
# - restricted HCP data
#
# Generates:
# - subject_ids.txt
# - palm_permutations.txt

addpath("~/Downloads");  # folder containing "hcp2blocks"
addpath("~/Downloads/PALM");  # folder containing PALM

rand('state', 0);

subject_ids = textread('~/Private/HCP/hcp_fMRI_subject_ids.txt');

EB = hcp2blocks('~/Private/HCP/RESTRICTED.csv', 'EB.csv', false, subject_ids);
[Pset, VG] = palm_quickperms([], EB, 1001);

dlmwrite('subject_ids.txt', EB(:, end));
dlmwrite('palm_permutations.txt', Pset);
