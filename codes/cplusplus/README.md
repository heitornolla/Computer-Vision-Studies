# Should you learn C++?

Sure! Why not?

I wouldn't prioritize it, unless you1re in a very specific situation where you need it more than anything else, but C++ is a quite important language to learn. Pretty much anything you'll ever code runs in C++ either under-the-hood (e.g. Python, Pytorch...) or directly (e.g. deploying a .pth model with C++).

I'm dedicating some time to learn C++ myself and I'll document resources and useful codes here. For now, I'm going through [this](https://www.learncpp.com/) website, which is going pretty well so far. It starts from absolute scratch, so it may be best to just skim over some of the initial chapters if you're more experienced.

"C++ in PyTorch" covers two very different workflows: (i) writing a small piece of fast C++ code and using it inside Python, or (ii) taking a finished Python model and running it entirely in C++ (no Python installed).

As a roadmap, I've defined for myself two "phases". In Phase 1, I look to speed up a specific part of Python code by rewriting just that part in C++. Phase 2 will then remove Python entirely.

Some useful documentation for these experiments is the [C++ Pytorch](https://pytorch.org/cppdocs/).

I'm using C++ with WSL and have set it up C++ with [this](https://code.visualstudio.com/docs/cpp/config-wsl) tutorial.

I'm using LibTorch 2.9.1 with CUDA 13.0.

--

I'm now going through [this](https://github.com/PacktPublishing/Hands-On-Machine-Learning-with-CPP) Packt book which is quite good. Definetly recommend!
