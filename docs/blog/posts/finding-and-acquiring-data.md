---
draft: false
authors: [raynold]
date: 2023-08-15 
categories:
  - Deep Learning
  - Data
---
# Finding & Acquiring Data: The Cornerstone of Magical Models 🧙‍♂️🔍

Welcome back! Deep Learning sorcerers of the future! 🌌

If deep learning was a spell, then data would be it's most powerful ingredient. 📜✨ While in our previous chapter, we gave ourselves the right tools and spells, this chapter will guide us into the heart of every magic model: **data**.

<!-- more -->

Imagine trying to cast a spell, without it's magic essence. 🚫🔮 That's just the same as trying to cast a deep learning spell without data. In both scenarios, our trials would fail. For any deep learning spell to work, they need data. 💧🧠

Join me as we go on a expedition across the great halls filled with scrolls of knowledge. 
We will go from the chambers of `Kaggle` to the shelves of `UCI`. 🏰🗺️ Together, we wont only learn to find these scrolls, but also to understand and choose the ones that fit our goal.

## The Essence of Data in Deep Learning

### Understanding the Foundation 🏰

Deep Learning is a powerful combination between algorithsm and computations. But remove all these spells and you will discover that the heart of this magic is, and always will be, **data**. 📜✨

Imagine a young wizard, eager to learn a new spell. Without a spellbook to guid him, or a mentor to show the way, he would be walking a **very** difficult path. In the world of deep learning, data acts as that all important spellbook, guiding our models to success.

**Why is data so important:**

1. **Training Ground:**

    Just like a newbie learns by practicing, our models learn by studying data. Each piece of data acts like a puzzle 🧩, teaching the model bit by bit, until it can see the larger picture.

2. **Validation and Testing:**

    The moment a model is trained, it's like a wizard ready to face the challenge. Testing it with new data makes sure that our spells are effective, not just inside our academy, but in the whole world beyond 🌍

3. **Continous Improvement:**

    Magic, just like technology, grows stronger every day. By always giving new data to our models, we make sure they adept and grow, becoming better with each spell cast. 🌱➡️🌳

**The Balance of Quanity vs. Quality: How much is enough?:**

While it might be tempting to gather so much data it can fill a ocean (trust me, i've been there), hoping it will make our spells invincible, it's very important to strike a good balance.

- **Quality:**

    A million scrolls won't help, if they are filled with nonsense. Quality data makes sure our models are learning the right lessons. It's better to have a small bit of accurate and relevant data, than heaps of irrelevant ones.

- **Quantity:**

    While quality is very important, quantity cannot be underestimated. A single spellbook might teach the basics, but a big library allows for a deeper understanding and mastery.

    To summerize, data is to deep learning what essence is to magic. And as you will discover, finding the right blend of data is a great journey in itself! 🌟

## Navigating the Grand Halls of Data 🌍🔍

Walking through the world of deep learning, you will soon realize the huge amount of different types of data available. These grand halls are filled with scrolls and books, all of them representing unique datasets waiting to be used. But how do you decide which path to take and which scroll to use?

### The Importance of Choosing the Right Data 🗝️📜

Every project or spell you want to cast, has it's own requirements. Choosing the right data is the same as choosing the right ingredients for a potion. Making the wrong choice, and you might end up with weird results, or even worse, a spell that hits you o.O!

1. **Relevance:**

    Make sure that the data fits your problem. If you are trying to understand the song of birds 🐦, analyzing scrolls of dragon lore 🐲 won't be of much help...

2. **Diversity:**

    The more diverse your data is, the stronger your model will be. It's like fighting different types of enemies, each of them will add a different type of experience to your power.

3. **Freshness:**

    In the vast evolving world of magic and technology, old data, can lead to outdated solutions. Always be looking for the latest and greatest scrolls or updates to your datasets.

### Discovering Great Repositories 📚🌌

Luckily, for young wizards like us, many great wizards of the past have put together their knowledge and made them available in big repositories. Here are a few of them where you can begin your quest:

1. **Kaggle:**

    Often called the 'grand arena' of data sciense, Kaggle not only offers you lots of datasets, but also challenges and competitions to test your skills.

2. **UCI Machine Learning Repository:**

    A very trusted treasure trove of datasets, curated and maintained by the scholars of the University of California, Irvine.

3. **Datasets from famous institutions:**

    Many magic schools 🏰 like MIT, Stanford, and Harvard release datasets for the public to use. Make sure to keep an eye out on their releases and announcements.

## The Adventure of Acquiring Data 🌄🎒

Going on a quest on search for data, is much like going on a great adventure. There are maps to follow, challenges to face, and treasures to discover. But don't worry, with the right compass and guidance, you will find the hidden gems that will make your deep learning spells stronger.

### Reading the Map: Identifying Data Needs 🗺️🔍

Before diving in the sea of data, it's important to know, exactly what you are looking for.

1. **Purpose and Your Goal:**

    Write down the core goal of your deep learning model. If you're crafting a potion to heal plants 🌱, you won't be gathering ingredients from the depths of the ocean.

2. **Features and Attributes:**

    Write down all the specifics. What kind of data points, characteristics, and information do you need? This will help you narrow down your search, to make sure you gather only what's important.

3. **Data Format:**

    Be clear about the format you need. Are you looking for images 🖼️, text 📜, audio 🎵, or a combination?

### Going on the Quest 🥾🌲

Now knowing what you are looking, it is time to start the journey into the world of data.

1. **Public Repositories:**

    As mentioned before, platforms like Kaggle and UCI are your go to places. Offering a wide variety of datasets that have been processed and prepared, start your quest here!

2. **Web Scraping:**

    Sometimes the data you need lays deep in the world wide web, you can go out and acquire this data by hand, or if there's too much data to get by hand, tools like `Beautiful Soup` or `Scrapy`, can help you automate this process

3. **APIs:**

    Many platforms and services also offer APIs that allow you to access and retrieve data.
    For example the [Google Search API](https://developers.google.com/custom-search/v1/overview) allows to scrape the search results of google and learn about many websites that Google has discovered.

## Creating our Dataset: Crafting the Beat of Magic 🥁🎶

Going into the world of deep learning, there is this magical feeling, when you create magic based on your own dataset. It's like crafting a wand from a tree you have grown yourself. The connection is deep, and the spells are truly powerful. You will know the dataset by heart, which makes working with it, so much better.

In this section, let's find out how to create our own dataset, filled with samples of the drums.

### Assembling our Ingredients: Getting Sample Packs 📦✨

Every great potion, starts with the best ingredients. For our drum magic, we began by purchasing several high-quality sample packs. These packs are our treasure chests 🎁, filled with samples waiting to be uncovered.

As a producer, you will probably already have many samples in your library, use those!

### Laying Down the Foundations: Dataset Structure 🏰📂

To keep our magical ingredients organized, we will create a clear structure for our dataset.
In your file explorer, in the folder we created for our project, create several new folders using the following structure:

```plaintext
dataset
├───drums
    ├───claps
    ├───clicks
    ├───cymbals
    ├───hats_closed
    ├───hats_open
    ├───kicks
    ├───percussion
    ├───rides
    ├───rimshots
    ├───shakers
    ├───snaps
    ├───snares
    ├───tambourines
    └───toms
```

This structure will serve as our library 📚, with each folder dedicated to a specific type of sound.

### Sorting the Magic: Organizing Samples 🗄️🔮

With our foundation created, it is time to dive in to our purchased sample packs and begin the big task of organization. The same as sorting mystical herbs, each sample is carefully listened to and placed in the right folder.

- Kick samples go into the `kicks` folder.
- Snare samples will find their way in the `snares` folder
- The crisp sound of closed hats go to `hats_closed`
- And so on, for all type of audio samples

It's important that each folder is nicely balanced.
It's better to have 100 kicks and 100 toms, then 1000 kicks and 10 toms. 

If you don't have enough samples for each type, make sure to remove the folders with not enough samples. Our sample classifier, won't be able to classify this type of sample anyway, so it's best not to confuse it.

Be patient and precise during this task, a well-organized dataset is like a well-organized spellbook. It creates efficiency, accuracy, and a touch of pure magic.

* Because our dataset contains premium samples, we unfortunately can't share this dataset publicly. A small dataset you could start with, is [Kaggle's Drum Kit Sound Samples](https://www.kaggle.com/datasets/anubhavchhabra/drum-kit-sound-samples)

### Verification & Refinement 🧙‍♂️🔍

When all samples are in place, we do a deep verification. Making sure all our ingredients are of the highest quality.
We listen to the samples, checking for misplacements, and refining the organization.

## Peering into our Chosen Scroll

Every great wizards knows the importance of understanding the scrolls and spellbooks they use. Like this, before we dive in classifying samples, it's important to understand our data. This is one of the reasons i like building my own datasets, while it's a lot of work, it makes me understand the data on a deeper level.

Besides that, even creating your own dataset, will usually still leave holes in your understanding of the data. So let's look at some ways that we can use to help us understand the data, it's structure and possible challenges we might need to overcome.

### Setup ⚙️
Let's go look at our data in a bit more technical way.

- **Installing Librosa:**

    To interact with our audio sample programatically we use a great library for python called `librosa`. This is a package used for music and audio analysis.

    Go into your terminal/command prompt, activate our magical room (virtual environment) and run:

    ```bash
    pip install librosa
    ```

- **Installing Seaborn:**

    Seaborn is a beautiful data visualization package based on [matplotlib](https://matplotlib.org/), it gives you a easy interface to work with to draw attractive and informative graphics:

    Run the following in your command prompt/terminal:

    ```bash
    pip install seaborn
    ```

- **Open up a new Jupyter notebook:**

    In the terminal/command prompt run:

    ```bash
    jupyter notebook
    ```

- **Make a new notebook:**

    Click on `New` in the top right of the user interface.
    Choose `Notebook`

- **Give it a proper name:**

    I like to name the notebooks i create, so in the future i know exactly which one is which.

    Choose `File` and click `Rename...`
    Name it something that you can identify it from. I called it `analyzation.ipynb`

### Summoning our first sound: Audio Playback 🎧🪄

Before we dive in the technical stuff, let's take a listen to some of our samples.

PS: Make sure to select a .wav file, i've tried several .aif files and wasn't able to get this to playback in this section (don't worry, they still work fine for our AI training).

Enter the following in our notebook:

```python
import IPython.display as ipd

ipd.Audio('path-to-sample') 
# Where path-to-sample is to a audio file in our dataset for example:
# ipd.Audio('dataset/drums/kicks/KSHMR Big Kick 01 (D).wav')
```

If u run the above cell, you can playback the chosen sample.

### The heartbeat of our samples: Visualizing Waveforms 📈🌊

Visualizing the waveform of some of our samples from all the categories can give us some insights into how their amplitude patterns change over time.

Create a new cell and enter the following:

```python
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load('path-to-sample')

plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title('Waveform')
```

### Understanding our durations ⏳📊

Let's take a look at the distribution of the duration of the samples in our dataset. This way we can see if we have any outliers and it will help us set a standard length for our model.

Let's first create a helper function, which scans our dataset and returns the filepath for each of our samples.

Create a new cell that will hold our function and enter:

```python
import os

def get_sample_paths(root):
    """Recursively gather all audio sample paths within our dataset directory."""
    sample_paths = []

    for dir, names, files in os.walk(root):
        for file in files:
            if file.endswith('.wav') or file.endswith('.aif'): # Assuming WAV/AIF format, but you can modify or extend as required
                sample_paths.append(os.path.join(dir, file))

    return sample_paths
```

Now let's use this function gather our samples and then we can look at the duration of each sample.

Create a new cell with the following code:

```python
import librosa
import seaborn as sns

sample_paths = get_sample_paths('dataset')

# Get the durations for our dataset
durations = [librosa.get_duration(path=f) for f in sample_paths]
durations_ms = [i * 1000 for i in durations] # Convert our duration into milliseconds, for easier analyzation

# Visualizing the distrubtion of our durations
sns.histplot(durations_ms, bins=20, kde=True)
plt.title('Duration Distribution')
plt.xlabel('Milliseconds')
plt.ylabel('Number of Samples')
```

When we run the above cell, we get the following distribution for our dataset:
![Duration Distribution](/assets/images/finding-and-acquiring-data/duration_distribution.png)

We can now see that the majority of our samples are around the 1000 ms durations.

### Dissecting the Frequency Domain: Spectogram Analysis 🌌📊

The spectogram is a beautiful representation of audio in a visual way. It shows the frequencies of a audio file as they vary with time. This in my opinion is one of the most important visualizations to understand how a sound behaves.

```python
import numpy as np

# Picking the first magical audio scroll from our collection
first_sample = sample_paths[0]

# Unrolling the scroll to hear its sounds and mysteries
# 'y' is the melody we hear
# 'sr' is how fast the scroll sings its song
y, sr = librosa.load(first_sample, sr=None)

# Making our canvas big enough to paint the magic on
plt.figure(figsize=(10, 4))

# Turning the scroll's song into a beautiful picture
# We use a spell to capture the song's energy
# Then, we color it using the power of decibels
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

# Painting the song on our canvas
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')

# Adding a magical color guide to our painting
plt.colorbar(format='%+2.0f dB')

# Giving our masterpiece a name
plt.title(f'The Magic of {os.path.basename(first_sample)}')

# Revealing our magical art to the world
plt.show()
```

When we run this, we get the following spectogram:
![Sample Spectogram](/assets/images/finding-and-acquiring-data/124_Dance_Clap_01-Spectogram.png)

### Our Audio Type Distribution: Class Sample Counts 🎲📋

It's important to know the number of samples for each audio type. This so we can make sure that our samples are evenly spread across all types.

Run the following in a new cell:

```python
import os

# Setting the path to the main dataset folder
dataset_path = "dataset/drums"

# Using a spell to gather all the magical categories from the kingdom of 'dataset'
categories = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

sample_counts = {category: len(os.listdir(os.path.join(dataset_path, category))) for category in categories}

sample_counts
```

Our sample counts look like this (as you can see it's not properly balanced, we still have some work to do to make our dataset better (always be improving!)):

```json
{
  'claps': 425,
  'click': 10,
  'cymbals': 168,
  'hats_closed': 104,
  'hats_open': 161,
  'kicks': 322,
  'percussion': 0,
  'rides': 140,
  'rimshot': 30,
  'shakers': 56,
  'snaps': 33,
  'snares': 463,
  'tambourines': 0,
  'toms': 81
}
```

We have done some great analyzing! We have a deeper understanding of our data, but we also made some scripts, which can also be used in the rest of our journey!

Remember, the key in mastering deep learning lies in understanding the data you work with! 🧙‍♂️📜🎶

## Conclusion

With our machine properly setup and our data ready, we can now finally begin with model training (YAY! 🥳)