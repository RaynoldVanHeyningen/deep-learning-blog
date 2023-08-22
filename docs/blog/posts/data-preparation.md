---
draft: false
authors: [raynold]
date: 2023-08-22 
categories:
  - Deep Learning
  - Data
---
# Preparing the Data: Opening our Toy Box 🎁👀

Imagine each sound as a beautiful toy in our toy box. We have lots of them, from kicks to cymbals. But how do we show these sounds to our computer..?

By turning them into pictures!

<!-- more -->

### Picture Magic: Turning Sounds into Images 🎵➡️🖼️

Have you ever thrown a stone into the water and watch the ripples?
That's also how sound works. We are going to take those ripples and turn them into pictures called a `spectogram`.

## Preparing our Toys: Getting Them Ready to Play! 🧸🚿

### Making Sound Pictures: Yes.. we're back to crafting at school 🫣🎨🖌️

**Why do we make pictures of sound?**

One of the easy and well researched ways of working with AI is using images, this is because images are easy for computers to understand. That's why we will turn our sounds into colorful pictures called `spectograms` (just like we did when analyzing one of our samples in the last lesson).


**Installing TQDM**

Now, because turning our sounds into pictures, takes a pretty long time. It seems very helpful to me to add a progress bar. Just so we know something is happening...

In your command prompt / terminal, run:

```console
pip install tqdm
```

**Let's create a new notebook!**

Whisper the incantation into your terminal or command prompt:

```bash
jupyter notebook
```

**Create a fresh Canvas:**

Upon the scroll in the top-right corner, click on `New`.
From the magical list, select `Notebook`.

**Name Your Magical Book 📖✨:**

Click on `File`, and choose the enchantment `Rename...`. Give it a name that fits with our current quest. How about `SoundToPainting.ipynb`?

#### Gathering our sounds

Like in the previous section, we will first be gather the locations of all our sounds in our dataset, let's use the same code in our first cell:

```python
# First we need to import the "os" spellbook, so we can navigate our file system
import os

# Here, we create a spell to gather all our sound scrolls (audio files).
def gather_our_sounds(dataset_directory):
  """Recursively gather all audio sample paths within our dataset directory."""
  # We will keep all the found scrolls here.
  sample_paths = []

  # We walk through all the places (folders) in our dataset.
  for dir, names, files in os.walk(dataset_directory):
    # For each scroll (file) we find...
    for file in files:

      # We check if the scroll sings in 'wav' or 'aif' tune.
      if file.endswith('.wav') or file.endswith('.aif'): # Assuming WAV/AIF format, but you can modify or extend as required

        # If it does, we add it to our collection.
        sample_paths.append(os.path.join(dir, file))

  # We return our collection of magical sounds to the code that is calling this function.
  return sample_paths
```

Now let's see how this looks like, in the new cell, let's use this function and display the paths:

```python
all_sounds = gather_our_sounds('dataset')
all_sounds
```

It will show a list of all your samples in the dataset. Here's a small section of how my dataset looks like:

```python
[
  'dataset\\drums\\claps\\IS_124_Dance_Clap_01.wav',
  'dataset\\drums\\claps\\IS_124_Dance_Clap_02.wav',
  'dataset\\drums\\claps\\IS_124_Dance_Clap_03.wav',
  'dataset\\drums\\claps\\IS_124_Dance_Clap_04.wav',
  'dataset\\drums\\claps\\IS_124_Dance_Clap_05.wav',
  'dataset\\drums\\claps\\IS_124_Dance_Clap_06.wav',
  'dataset\\drums\\claps\\IS_124_Dance_Clap_07.wav',
  'dataset\\drums\\claps\\IS_124_Dance_Clap_08.wav',
  'dataset\\drums\\claps\\IS_124_Dance_Clap_09.wav',
  'dataset\\drums\\claps\\IS_124_Dance_Clap_10.wav',
]
```

#### Turning our sounds into beautiful pictures

Let's now use this list of sounds and turn them into spectograms.

We loop through all our sounds and create spectograms for them. We store them together with the audio file, so we can keep the same directory structure.

We will be running the following process in batches, if we would process our entire dataset at once, well let's just say... our PC wouldn't be too happy. I've even experienced the infamous blue screen of death, because i had too many samples loaded in memory.

Since the more samples we have, the more that can go wrong during processing. This is why we also add a check if the spectogram already exists, if it already exists, it will skip processing that file.

Imagine processing for 10 minutes, and then jupyter crashes >.<, if we then rerun the cell, we don't have to reprocess all the files we already did...

Or adding new files to our dataset and we want to just preprocess our new samples.

Create a new cell, and enter the following:

```python
import librosa
import numpy as np
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm  # Importing tqdm.notebook.tqdm for a cleaner display in Jupyter
import gc # Garbage collection

# Let's keep our magic efficient by processing in smaller batches
batch_size = 100

# Create an outer progress bar for the entire dataset
outer_pbar = tqdm(range(0, len(all_sounds), batch_size), desc='Overall Progress')

for i in outer_pbar:
  # Create an inner progress bar for the current batch
  inner_pbar = tqdm(all_sounds[i:i+batch_size], desc=f'Processing batch {i//batch_size + 1}', leave=False)
    
  # Load and process the files in batches of 'batch_size'
  for sound in inner_pbar:
    exists = False

    # Determine the name of the spectogram file
    dest = sound.replace('.wav', '').replace('.aif', '') + '-Spectogram.png'
    
    # Check if the spectogram file already exists
    if os.path.exists(dest):
        # If it exists, skip this iteration
        exists = True
        continue

    # Load our sound into librosa
    melody, speed = librosa.load(sound, sr=None)

    # Use a spell from librosa to turn our melody into a beautiful spectogram
    spectogram = librosa.amplitude_to_db(np.abs(librosa.stft(melody)), ref=np.max)

    # Prepare our canvas to paint on
    plt.figure(figsize=(10, 4))

    # Paint the magic on our canvas
    librosa.display.specshow(spectogram, sr=speed, x_axis='time', y_axis='log')

    # Painting a decibel meter
    plt.colorbar(format='%+2.0f dB')

    # Saving our spectogram
    plt.savefig(dest)

    plt.close() # This makes sure we don't show the image in the notebook, since we just want to save it in our dataset.

  if not exists:
    # Now let's free up our memory!
    del melody, spectogram, speed
    gc.collect() # Forces garbage collection to collect garbage
```

Now when we run this cell, we can see a beautiful progress bar, keeping track of saving our spectograms:

![Alt text](/assets/images/data-preparation/progress.png)

## Setting The Stage For Our Grand Spell 🪄🎵

With our sounds now painted as beautiful images, we are ready to continue our journey.
In the next chapter, we are becoming spellcasters! Using our painted sounds, we are going to create a grand spell --a model--, that is going to predict our future sounds.

Are you ready..? 🎶🔮
