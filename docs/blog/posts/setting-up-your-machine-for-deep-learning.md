---
draft: false
authors: [raynold]
date: 2023-08-12 
categories:
  - Deep Learning
  - Setting up
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

While we are getting deeper into the deep learning world, we need to equip ourselves with the right AI tools. Just like a wizard has their favorite wand, in deep learning, we have several libraries we can use. Each have their own unique capabilities and strenghts.

### Let's look into the chamber of libraries

Before we dive in, let's look at the different tools we can use:

- **TensorFlow 🌊:**  
Created by the wizards from Google, TensorFlow is very versatile and widely used. It gives you many high-level and low-level tools, this way it's usable for everyone, from beginner to the most powerful wizards.

- **PyTorch 🔥:**  
Coming from Facebook's (or Meta nowadays...) school of sorcery, PyTorch is like a spellbook that changes and adapts while you are casting spells. Being this flexible allows wizards to experiment as much as they please, making it loved by people who love to tinker and innovate in their magic.

- **FastAI 🌪️:**  
A rising star in this world of deep learning. FastAI is built on top of PyTorch. People celebrate it, because it's super user-friendly and being able to get world-class results with very little code. This library is the perfect example of a "magic spell" - being able to achieve great things with simple incantations.

There are many libraries to choose from, but for our journey, we will be using the great power of **FastAI**! Being able to achieve beautiful results very fast, and then dive deeper and use the power of PyTorch to tinker and innovate when we feel ready, is a power we can't pass up on.

### Installing FastAI: Equipping our Spellbook 📖

Because FastAI is built on top of PyTorch, we will first have to install and setup PyTorch. Just like gathering the base ingredients to create a powerful potion.

1. **Setting up PyTorch 🔥:** 

    While i will show you the current way to install PyTorch, make sure to check out [PyTorch's official installation guide](https://pytorch.org/get-started/locally/), since the world of AI changes fast, and the version we will currently be installing, might be replaced tomorrow.

    On the PyTorch installation guide, we have to choose between several options:

    - **PyTorch Build:**  
    Do you want to play with the newest features, which can have bugs, or do you want to work with a stable version?  
    We will choose `Stable`` (which is currently on version 2.0.1)

    - **Your OS:**  
    Choose the operating system you are using, we will be using `Windows`

    - **Package:**  
    This is where we choose, *how* we want to install PyTorch.
    Since we have been using `pip` so far, we will choose `Pip`

    - **Computer Platform:**  
    Here we choose what hardware we want to use with PyTorch.
    CUDA gives us the ability to use our GPU, while the CPU option
    will install PyTorch, with only CPU capabilities. Choosing
    CUDA is the best of both worlds, we can still use the CPU if
    we wanted. We will choose the latest supported CUDA version by our GPU, which currently is `CUDA 11.8`

    Having selected our options, the installation guide will show us what command to run to install PyTorch. The command which shows up for us currently is:

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

    So in our crystal ball (Command Prompt/Terminal) enter this exact command.

    *If you have closed the command prompt after the last steps, make sure to open it again, `cd` into the correct directory and `activate` our magic room as described above*

2. **Installing FastAI 🌪️:**  

    With PyTorch as our foundation, it is now time to summon `FastAI`. In your crystal ball, enter:

    ```bash
    pip install fastai
    ```

3. **Verifying our installation 🔍:**  

    To make sure that FastAI and PyTorch are ready to play with, we can run a quick test!

    In your crystal ball enter:
    ```python
    python -c "import fastai; print(fastai.__version__)"
    ```

    If everything went the way we want things to go, our crystal ball will tell you the version number of FastAI. If you see the version number, it is ready to be used for our deep learning quests!

## Testing Your Setup 🧪

Now that we have all our magic tools, let's make sure everything is working by checking if our GPU is setup correctly.

### Running a Simple Deep Learning Spell

1. **Preparing our spell:**

    Remember! Make sure your magic room is activated before we get started!

2. **Casting our GPU checker:**

    Run the following command:

    ```python
    python -c "from fastai.vision.all import *; print(torch.cuda.is_available())"
    ```

    This usefull spell asks FastAI and PyTorch to check if they can find and use the GPU. If they can, they should respond with `True`.

### Understanding the output 🔮

If you saw `True`, congratulations! 🎉, your GPU is ready to lend you it's powers!

If you saw `False`, don't worry. It might mean your system is not using GPU acceleration. This can be because of different reasons, maybe the GPU is not supported, or perhaps there is a slight misconfiguration somewhere.

For example i had this issue, because on the PyTorch website it uses `pip3` in the command. However, we have been using `pip` until now. Usually this should be fine, but if you have multiple python installation on your computer, both versions of pip could be pointing to different installations. Running the command using `pip` fixed it for me.

If your GPU might not be supported, you can always use cloud providers like `Google Colab` to borrow the power of the GPUs in the magical clouds.

Anyway, it's great practise to understand and test your tools. Doing this will make you know your wand and spellbook for when you plan to cast bigger and more complex spells!

## The Magical Book: Jupyter Notebooks 📖

Imagine having access to a book that reacts to you spells immediatly when you enter them, where you can draw diagrams, make notes, and even have conversations with. In the world of deep learning, this magical book is known as a Jupyter Notebook.

### What makes Jupyter Notebooks so special?

1. **Interactive Spells:**

Write a spell (or code) and see it's effects immediatly. This instant feedback makes it perfect for experimentation

2. **Blend of Text and Magic:**

Combine notes, diagrams and code in one place. It's like having a spellbook with annotations!

3. **Shareable:**

Once you have perfected a spell or discovered something new, you can easily share it with other wizards.

### How to get this magical book?

1. When you are in your virtual environment (or magical room), simply write:

```bash
pip install jupyter
```

2. To open the book and begin writing your spells, write:

```bash
jupyter notebook
```

A portal will open in your browser, which leads you to your Jupyter environment.

## Mastering Your Magical Book 🧙‍♂️📖

### Starting with a new notebook

Let's create a new notebook in our brand new environment:

- Click the `New` button on the top right
- Choose `Notebook`

Welcome to your first notebook!

### Cells - The Heartbeats of Jupyter

- Each notebook consists of cells. Each cell can contain code or written text (in `Markdown`)
- By default a cell will be of type `Code`, to test this out select the first cell and type in
```python
1 + 1
```
- To run a cell, click on it and press `Shift + Enter`. The result will be displayed right under the cell. Try it out and see the result of our very difficult calculation!
A new cell will be automatically created for you.
- To change a cell's type, use the dropdown menu at the top. Choose `Code` for Python code or `Markdown` for written text. Try it out now by selecting `Markdown` for the second cell

### Writing and Formatting Text:

- To write titles, use the hashtag (`#`). For example write, `# Title`, this will turn into a big title. `## Subtitle` will give a slightly smaller one, and so on.
- To make text bold, wrap it in double asterisks like `**this is bold**`.
- For bullet points, use dashes (`-`) or asterisks (`*`).

[Read more about markdown](https://itsfoss.com/markdown-guide/)

### Saving and Closing Your Notebook:

- To save your work, click on the floppy disk icon on the top left or press `Ctrl + S` (or `Cmd + S` on Mac).
- When done, click on the `File` menu and choose `Close and Shut Down Notebook` to shutdown the notebook. This ensures the spells (codes) you've been running also stop.

### Tips to have a GREAT Jupyter time:

- Autocomplete is your friend! While writing code, press `Tab` to autocomplete Python functions or variable names.
- If your notebook ever gets stuck, go to the `Kernel` menu and choose `Restart Kernel...`. This acts like a mgaical reset button, refreshing your environment.

### Venturing Further:

- Jupyter Notebooks have many built-in functionalities, like magic commands (commands starting with `%`). For example, `%time` before a spell (code) will show you how long it takes to run.
- You can also look at data, plot graphs, and even embed videos. The limits are endless!

Remember, Jupyter Notebooks are more then just a tool. It's your companion in this deep learning journey. The more you work with it, the better you will get!

## Conclusion 🎉

The journey to becoming a deep learning magician is enchanting but very complex, and today, we have taken a very big step preparing for our journey! Here's what we did today:

- **Introduction to this Magical Realm:**

We've look at the importance of GPUs in boosting our spells and looked at using CUDA to harness it's power

- **Setting up our Magical Language:**

With Python as our trusty spellbook, we are ready to cast our spells. We learned about the `pip` wand and created our own magic room, a virtual environment, to keep our spells organized.

- **The chamber of libraries:**

We looked at the rich world of deep learning libraries, looking at the benefits of each. We have chosen FastAI as our companion and we made sure it was equipped with the power of PyTorch.

- **Testing the waters:**

We tested our setup, to make sure all our powers are ready to use for our upcoming adventures.

## A Glimbse Beyond the Horizon 🌅

In our next post, we are going to look deeper into FastAI. We are going to look at deciphering the melodies and rhythms of the world around us by working on a audio sample classifier. Think about this, by the end of our next session, you will be able to play a drum sound to your computer - wether it's a kick or snare - and it will magically tell you what type of sound it is!

So keep your wands ready and spellbooks open. The world of deep learning is right around the corner!