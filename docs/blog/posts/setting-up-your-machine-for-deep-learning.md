---
draft: false 
date: 2022-01-31 
categories:
  - Hello
  - World
links:
  - Getting started with Insiders: insiders/getting-started.md#requirements
  - setup/setting-up-a-blog.md#built-in-blog-plugin
---
# Setting Up Your Machine For Deep Learning

Hey there, explorer! 🚀

Guess what? We are about to start a super fun and exciting adventure
into the world of deep learning. Now you might be wondering, “What is
deep learning?”.

Think of it as a magical tool that not only lets computers recognize
things like pictures of cats and dogs, but also helps them create new
sounds, melodies, and even whole songs! 🎵🐱🎶

Have you ever dreamt of a computer composing its own music, or changing
your voice to sound like your favorite cartoon character? With deep
learning, all of this becomes possible! 🎤🎧

But, just like setting up a new instrument before playing it, we need to
make sure our computer is ready for this new world.

This guide will help us tune our computer to perfection. We’ll get it
ready step by step, and i promise it will be super fun! 🎹🧙‍♂️

Are you excited to begin?

<!-- more -->

## Understanding the basics

### 🧠 CPU vs GPU

The CPU is the brain of the computer, it’s so important, that without
it, your computer would turn into a simple brick. It’s responsible for
all the main things a computer does, running windows, your internet
browser and your favorite DAW. Under the hood all it’s doing is
calculating numbers.

Now the GPU (Graphics Card) is responsible for making things look
pretty. It’s wahat you need to play your favorite games, like Diablo,
Fifa, Minecraft and all other games you play.

### Why do i need a GPU

The way the GPU is created, is that it’s super fast at calculating the
numbers that show you the graphics, that simulate physics etc. And
because it’s so fast at these tasks. AI experts have found out that
using your GPU to train your AI application, is alot faster then using
your CPU.

Now if you don’t have a good (or supported) graphics card, you can still
use your CPU, however training your AI will take alot longer.

## Checking Your Hardware

### 🕵️ Become a Computer Detective!

Before getting started with deep learning, we have to become a bit of a
detective and take a look inside our computer. It’s just like checking
your backpack before you go camping, you want to make sure you have all
the essentials!

### 🖥️ Do I have a deep learning-friendly GPU?

1.  **Windows:**

-   Click on the search bar at the bottom
-   Type in `Device Manager` and click on it.
    -   Look for `Display adapters` and click on the arrow next to it.
        If you see names like `NVIDIA` or `AMD`, you might have a
        compatible GPU! 🌟

1.  **Mac:**
    -   Click on the Apple logo on the top left.
    -   Select `About This Mac`
    -   Look under `Graphics`. If you see `AMD` or `Radean`, that’s a
        possible GPU! 🎉
2.  **Linux:**
    -   Open your terminal
    -   Type `lspci | grep VGA` and press Enter.
    -   Names like `NVIDIA`, `AMD`, or `Radeon` mean you might have a
        compatible GPU! 🎈

### 🚀 Is My GPU Ready for Deep Learning?

Think of your GPU like a superhero. It might be ready to fly and save
the world, or it might still be training to get there. In the industry
the widely supported GPUs are `NVIDIA` GPUs. Deep Learning Frameworks
like `PyTorch` and `TensorFlow` mainly support `NVIDIA`, while it is
possible for `TensorFlow` to use `AMD` GPUs, it will be a very big task
to get this working.

Check out some reviews on your GPU related to deep learning tasks. Keep
an eye out for things like `CUDA cores` (These are the magic units that
help with AI), and how new your GPU is.

Newer models usually come with more powers!

### 💭 What if I Don’t Have a Hero GPU?

Don’t worry! Even if your GPU isn’t that superhero for deep learning, or
if you don’t have a GPU, you can still train AI models! There are many
magic clouds up in the sky, where alot of powerful GPUs live. These
places let you use their superhero GPU for a while, for a small price or
often even for free!

Here are some to look at: 

1. **[Google Colab](https://colab.google/):**
This is like a magic playground. It gives you a notebook (a magic book
where you can write code and see results). The best part? It comes with
a free GPU! Just remember, if you play for too long, the playground
might ask you to take a break. 
2. **[Paperspace](https://www.paperspace.com/):**
This is a realm where you can summon powerful computers with strong GPUs. You can use them for
a few coins, but they are super fast and worth it! 
3. **[Kaggle Notebooks](https://www.kaggle.com/code):** 
This place is like a training ground for wizards and witches wanting to experiment. They also
offer free GPUs. You can practice, learn, and share with other learners
here.

### 🔍 Quick Recap

-   Check your GPU brand.
-   See if it’s a superhero-level GPU for deep learning.
-   If not, visit one of the magic clouds to borrow their power.

In the next section, we’ll look at setting up the magic spells
(software) we need to start our deep learning journey! 🌌🔮

## Getting the Necessary Software

Before we can cast these deep learning spells, we need to gather the
correct ingredients. Think of them as the potions and scrolls needed by
wizards before casting their spells

### 🧙‍♂️ CUDA: The Magical Portal Between Worlds

Imagine a big magical world, where every sparkle ✨ of magic is a small
calculation. This realm is huge and beautiful, with millions of sparkles
every second. Now, think of a portal 🌀 that allows wizards to travel
between our world, and this magical world. Whenever they are in the
magical world, they can use all these tiny sparkles of magic. This
portal is what we call `CUDA` 💡. Which officialy stands for
`Compute Unified Device Architecture`, yes.. i know.. i didn’t think of
it either.

In the world of computers, billions of calculations happen every second,
especially in deep learning and AI 🤖. The main thing in deep learning
is crunching huge amounts of data 📊, looking for pattners,
understanding images 🖼️, recognizing a voice and even generating new
music 🎶! This requires soo much power.

**So, what about this `CUDA`?** 🧐

`CUDA` is a magical language and set of tools, created by `NVIDIA`,
which allows our software (the spells we cast ✨) to communicate
directly with `NVIDIA GPUs`. With `CUDA` we can tell the GPU precisely,
which calculations to perform and in what order, so we can truly
maximize it’s power 🚀.

Just think of it like this: `CUDA` helps our computer and GPU talk to
each other so they can work together to be more powerful (Gogeta vs
Vegito anyone? 🥊)

Without `CUDA`, our journey into deep learning would be like trying to
go mining in those pesky Minecraft caves. With `CUDA`, it’s like we’ve
got an enchanted pickaxe ⛏️✨, that lights up the caves when we are
digging. Wouldn’t we all want a pickaxe like that? 😄

### 👁‍🗨 Checking CUDA compatibility with your GPU

To use this powerful tool, our GPU needs to be able to read this CUDA
spellbook. Most newer GPUs are trained in this art, but to make sure,
you can visit the [NVIDIA CUDA GPUs
page](https://developer.nvidia.com/cuda-gpus). If you see your GPU
listed here, you are fully ready for this magical journey!

### 🔮 Installing CUDA

Now how do we get this beautiful spellbook?

That’s the beauty… we don’t need to!

In a future step, where we will be installing a popular deep learning
library, `CUDA` will be installed for you 💝.

How easy is that?!

## Setting up Python 🐍

🚀 Things are starting to get exciting! Remember how wizards use special
words to cast spells? Well, in our deep learning adventure, we will be
using a magical language called `Python` 🐍 to write our very own spells
🤯 (which are like computer programs!).

Just like how every magician needs to know special words to create
magic, we need to setup `Python` to start creating our deep learning
spells.

Let’s dive in and get our magic toolkit ready! 🎩✨

### The Magic of Virtual Environments 🌍✨

Have you ever seen a magician pull a rabat out of a hat 🐰? or make
things float in a specififc magic room? Now imagine if a magician would
be doing all their tricks in that same room. Things would get very
messy… full of floating rabbits and stuff 😂 (kinda cool though…)

This is where virtual environments come in. Think of these as special
magic rooms, where we can practice a specific spell, without affecting
all the other spells we might have.

By using these rooms, we make sure our magic spells (or code..) don’t
interfere with eachother and always work the way we want them to.

In the coding world, this makes sure that all the different projects we
will have, can have their own settings, their own libraries, and even
their own version of Python, without messing up the other projects.

To create a virtual environment, we will first need our basic version of
Python installed (which we can then use to create these magic rooms).

### Setting up our magic language: Python (with `pip`)

#### What the hell is pip 😵‍💫?

Imagine you’re a wizard, you’re progressing and you’re discovering there
are so many spells, more then what u already know. How do add all these
beautiful spells to your spellbook?

This is where `pip` comes in!

It’s like a magical scroll that let’s you add more and more spells (or
tools) that other wizards have created, all to our own spellbook!

Want to play with images or sounds? Ask `pip` to bring you the right
spells! (or in the real world, the library).

Now how do we set all this up?

#### How to install

1.  Go to [Python’s official website](https://www.python.org/downloads/)
    and download the latest version.
2.  Open the installer you download, and follow the instructions. \*Make
    sure to check the box that says `Add Python to PATH` during
    installation (this so we can use python from anywhere on our PC)

Now let’s make sure everything. To do that we will need to start using
the command line.

### The wizards crystal ball 🔮

Before we check our Python installation, we need to get to know one of
the most important tools for a wizard: The Crystal Ball, or in
professional terms - the Command Line (for Windows) or the Termianl (for
Mac and Linux).

#### What is the Command Line/Terminal?

Imagine you have a crystal ball. Whenever you ask it something, you’re
actually giving it a command, and the crystal ball executes this command
for you instantly.

The command line or terminal is exactly like that crystall ball! It’s a
place where you enter commands, and the computer immediatly does what
you tell it to.

#### How to access your crystal ball

-   **Windows Users 🪟**
    
    Press the `Windows + R` keys together, a small
    window will pop up. Type `cmd` and press Enter. This will open up
    the command Line.

-   **Mac Users 🍏**

    Press `Command + Space` to open up Spotlight. Type
    `terminal` and press Enter.

-   **Linux Users 🐧**

    This varies a bit depending on which
    distribution u use, but usually, you can find the Terminal in your
    applications menu or using your keyboard shortcuts like
    `ctrl + alt + t`.

### Checking our Python using our Crystal Ball

Now that we know how to open our crystal ball, we can give it some
commands to check out if Python is succesfully installed:

1.  Open your crystal ball
2.  Type `python --version` and press Enter
3.  The crystal ball should tell you which version of Python you have
    installed. If it does, that means your Python spellbook, is ready to
    work! If not, it means your Python was not installed correctly. Try
    following the Python installation steps again, or leave a
    comment/email and i will try my best to help you.

    ```console
    C:\Users\rvanh>python --version
    Python 3.10.11
    ```

#### Checking our pip install

On the same note as checking for Python, we can check for `pip`.

1.  In the same crystal ball type in: `pip --version`

    ```console
    C:\Users\rvanh>pip --version
    pip 23.1.2 from C:\Python310\lib\site-packages\pip (python 3.10)
    ```

If it shows a version number, you are good to go!

## Creating our first Virtual Environment

Setting up our first magic room (or virtual environment) is super easy!

Open your explorer, and create a folder on your computer where you
would like to store all your deep learning projects.  *Mine is
located in `D:\Development\AI`*

1.  **Windows Users 🪟:**
    -   In your crystal ball, we have to go to the folder we just
        created, using a spell called `cd` - which stands for
        `change directory`. In my case i would type:

        ``` bash
        cd "D:\Development\AI"
        ```

    -   If the folder you have created is on a different drive then the
        default, in my case `C` drive, the windows command prompt can
        sometimes be a bit weird, in my situation it did change the
        directory succesfully, but it doesn’t display it yet:

        ```console hl_lines="2"
        cd "D:\Development\AI"
        ```

        In this case, we would need to tell our crystal ball to work in
        this different drive - which in my case is the `D` drive. We can
        simply do that by writing `[Drive Letter]:`, so i would enter:

        ```console
        D:
        ```

        and then press enter:
        ```console hl_lines="3"
        cd "D:\Development\AI"
        D:
        ```

    -   Since we are now in the folder that holds all our AI projects,
        let’s create a folder for our first project, which will be a
        “Sample Classifier (more on that later)”:
        ```console hl_lines="3"
        cd "D:\Development\AI"
        D:
        mkdir sample-classifier
        ```

    -   Now let's move into that directory:
        ```console hl_lines="4"
        cd "D:\Development\AI"
        D:
        mkdir sample-classifier
        cd sample-classifier
        ```
    
    -   Now that we are here, we can create our virtual environment, using the syntax:

        `python -m venv [name-of-environment]`

        Where [name-of-environment] can be anything you want it to be,  
        let's name ours `sample-classifier-env`

        ```console hl_lines="5"
        cd "D:\Development\AI"
        D:
        mkdir sample-classifier
        cd sample-classifier
        python -m venv sample-classifier-env
        ```

    -   Our new magical room has been created, now let's tell our crystal ball, that from now on we use this room!
        *Anytime you want to work on this project, make sure to execute the following step, so your computer knows which room to use*

        ```console hl_lines="6"
        cd "D:\Development\AI"
        D:
        mkdir sample-classifier
        cd sample-classifier
        python -m venv sample-classifier-env
        sample-classifier-env\Scripts\activate
        ```

2.  **Mac and Linux Users 🍏🐧**
    -   Tell your crystal ball to move to our newly created folder, using a spell called `cd`:

        ``` bash
        cd "/home/Development/AI"
        ```

    -   We are now in the place that will hold our AI projects,
        let’s make a seperate place for our first project, a
        “Sample Classifier (more on that later)”:
        ```bash hl_lines="2"
        cd "/home/Development/AI"
        mkdir sample-classifier
        ```

    -   Let's go into that directory:
        ```bash hl_lines="3"
        cd "/home/Development/AI"
        mkdir sample-classifier
        cd sample-classifier
        ```
    
    -   Now we can create our virtual environment, the following syntax is used for creating a environment:

        `python -m venv [name-of-environment]`

        [name-of-environment] can be any name you want,  
        We named it `sample-classifier-env`

        ```bash hl_lines="4"
        cd "/home/Development/AI"
        mkdir sample-classifier
        cd sample-classifier
        python -m venv sample-classifier-env        
        ```

    -   Our room has been created, now we need to use this room!
        *Anytime you want to work on this project, make sure to tell the computer which room to use*

        ```bash hl_lines="5"
        cd "/home/Development/AI"
        mkdir sample-classifier
        cd sample-classifier
        python -m venv sample-classifier-env
        source sample-classifier-env/bin/activate        
        ```

You will know your virtual environment is active when you see its name on the left side of your command line or terminal. This room will help keep all your magical deep learning experiments organized and separate!

Whenever you want to leave the circle, just type `deactivate` in your command line or terminal.

## Installing our Deep Learning library

