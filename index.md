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

/resources/recon/03_brass_orig.mp3

/resources/recon/03_brass_vae.mp3

/resources/recon/03_brass_vrnn_inst.mp3

Flute:

/resources/recon/13_flute_orig.mp3

/resources/recon/13_flute_vae.mp3

/resources/recon/13_flute_vrnn_inst.mp3

Guitar:

/resources/recon/12_guitar_orig.mp3

/resources/recon/12_guitar_vae.mp3

/resources/recon/12_guitar_vrnn_inst.mp3

Mallet:

/resources/recon/04_mallet_orig.mp3

/resources/recon/04_mallet_vae.mp3

/resources/recon/04_mallet_vrnn_inst.mp3

Reed:

/resources/recon/14_reed_orig.mp3

/resources/recon/14_reed_vae.mp3

/resources/recon/14_reed_vrnn_inst.mp3

Keyboard (accordion) :

/resources/recon/22_keyboard_orig.mp3

/resources/recon/22_keyboard_vae.mp3

/resources/recon/22_keyboard_vrnn_inst.mp3

Keyboard (piano) :

/resources/recon/31_keyboard_orig.mp3

/resources/recon/31_keyboard_vae.mp3

/resources/recon/31_keyboard_vrnn_inst.mp3


## Generation

### Instrument Conditioning

6 samples were generated for each conditioning instruments.
All samples are generated with harmonic oscillator conditioned by a fixed fundamental frequency of 440Hz.

Brass:

/resources/AttrInstBrass.mp3

Flute:

/resources/AttrInstFlute.mp3

Guitar:

/resources/AttrInstGuitar.mp3

Mallet:

/resources/AttrInstMallet.mp3

Keyboard:

/resources/AttrInstKeyboard.mp3

Reed:

/resources/AttrInstReed.mp3
