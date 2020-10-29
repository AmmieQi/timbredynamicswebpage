---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
---
# Sound examples

## Reconstruction

Samples were encoded and decoded by the model. 
Strange blips sometimes present at the end of notes are caused by CREPE not estimating fundamental frequency correctly. Samples are played in order of Original->VAE->VRNN-Inst. All models have around 500k~530k parameters.

Brass: 



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
