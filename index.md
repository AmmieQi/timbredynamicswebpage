---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
---
# Sound examples

## Reconstruction

Samples were encoded and decoded by the model. 
Strange blips sometimes present at the end of notes are caused by CREPE not estimating fundamental frequency correctly. Samples are in order of **Original->VAE->VRNN-Inst**. All models have around 500k~530k parameters.

Brass: 

../../assets/recon/03_brass_orig.mp3

../../assets/recon/03_brass_vae.mp3

../../assets/recon/03_brass_vrnn_inst.mp3

Flute:

../../assets/recon/13_flute_orig.mp3

../../assets/recon/13_flute_vae.mp3

../../assets/recon/13_flute_vrnn_inst.mp3

Guitar:

../../assets/recon/12_guitar_orig.mp3

../../assets/recon/12_guitar_vae.mp3

../../assets/recon/12_guitar_vrnn_inst.mp3

Mallet:

../../assets/recon/04_mallet_orig.mp3

../../assets/recon/04_mallet_vae.mp3

../../assets/recon/04_mallet_vrnn_inst.mp3

Reed:

../../assets/recon/14_reed_orig.mp3

../../assets/recon/14_reed_vae.mp3

../../assets/recon/14_reed_vrnn_inst.mp3

Keyboard (accordion) :

../../assets/recon/22_keyboard_orig.mp3

../../assets/recon/22_keyboard_vae.mp3

../../assets/recon/22_keyboard_vrnn_inst.mp3

Keyboard (piano) :

../../assets/recon/31_keyboard_orig.mp3

../../assets/recon/31_keyboard_vae.mp3

../../assets/recon/31_keyboard_vrnn_inst.mp3


## Generation

### Instrument Conditioning

6 samples were generated for each conditioning instruments.
All samples are generated with harmonic oscillator conditioned by a fixed fundamental frequency of 440Hz.

Brass:

../../assets/AttrInstBrass.mp3

Flute:

../../assets/AttrInstFlute.mp3

Guitar:

../../assets/AttrInstGuitar.mp3

Mallet:

../../assets/AttrInstMallet.mp3

Keyboard:

../../assets/AttrInstKeyboard.mp3

Reed:

../../assets/AttrInstReed.mp3
