---
title: "Notes on JAX" 
route: "jax-notes"
description: "List of notes I acquired from using JAX"
tools: ["JAX", "Python", "Machine Learning"]
---

## JAX
If you are looking for a Python library that can help you write fast and efficient code for different hardware platforms, you might want to check out JAX. JAX is a library that combines three powerful features: just in time compilation, Autograd and XLA.

Just in time compilation (JIT) means that JAX can compile your Python code on the fly and optimize it for the target device. This way, you don't have to write separate code for CPU, GPU, or TPU. You can use the same code and let JAX handle the compilation.

Autograd is a feature that allows you to automatically compute gradients of your functions. This is very useful for machine learning and optimization tasks, where you often need to calculate the derivatives of your loss function with respect to your parameters. With Autograd, you can write your functions in a natural way and let JAX take care of the differentiation.

XLA stands for Accelerated Linear Algebra, and it is a compiler that can generate highly optimized code for linear algebra operations. XLA can leverage the parallelism and memory hierarchy of modern hardware devices, such as GPUs and TPUs, and make your code run faster and more efficiently.

JAX is a library that can help you write code that can be compiled to run on a CPU, GPU, or TPU without changing the code. JAX stands for just-in-time compilation, Autograd, and XLA. XLA is super quick. If you want to learn more about JAX and how to use it, you can visit its official website or check out some tutorials and examples online. In my experience, the best way to learn JAX is through reading books about it, looking through the source code, and of course, reading the documentation.

## JAX notes
### JAX mutable states
Jax HATES hates mutable state, and thats how functional programming usually works, you try your best to not mutate state. try to keep everything stateless.
```python
import Jax as jnp
X = jnp.array([1, 2, 3, 4, 5])

# You can't do X[3] = 10, JAX will complain
# Instead you have to create a whole new array and assign it and this can be done as the following:
X = X.at[3].set(10)
```
In the above example, we created a new array in memory and re-assigned X to it. This approach is not very efficient and is one of the issues with functional programming. Functional programming has only recently gained more attention because back in the day memory was very expensive. Nowadays, we can carry 16 gigabytes of memory in our pockets, so memory is no longer an issue. Functional programming is getting more attention because it is very good for parallelism since it avoids side effects. Side effects come from mutating a global variable or a variable shared with other functions. When parallelizing code, if you are using a shared variable you must synchronize it by using mutex or semaphores. This is not the case with functional programming.

Functional programming is based on the simple premise that functions should not have side effects; they are considered evil in this paradigm. If a function has side effects we call it a procedure, so functions do not have side effects. A function that returns always the same result for the same input is called a pure function. In addition, determinism is a key aspect of functional programming. We cannot allow a function to trigger a side effect just by calling it.

### JIT explained
JIT compilation is a technique that improves the performance of programs written in high-level languages like Java or Python. JIT stands for Just-In-Time, which means that the code is compiled right before it is executed, instead of ahead of time. This has several advantages, such as:

- The compiler can optimize the code based on the current state of the program and the environment, such as the available memory, CPU, and operating system.
- The compiler can generate native machine code that is tailored for the specific hardware and platform, which can run faster than interpreted code or generic bytecode.
- The compiler can avoid compiling parts of the code that are never used or executed, which saves time and space.

JIT compilation is not a new concept, but it has become more popular and widely used in recent years, especially for dynamic languages that have features like reflection, dynamic typing, and polymorphism. These features make it harder to compile the code statically, because the compiler cannot know in advance what types of objects or methods will be used at runtime. JIT compilation solves this problem by compiling the code on demand, when the types and values are known.

JIT compilation is not a silver bullet, though. It also has some drawbacks, such as:

- The compilation process itself takes time and resources, which can cause delays or overhead during the execution of the program. This is especially noticeable for short-running programs or scripts that do not benefit much from the optimizations.
- The compiled code may not be portable or compatible with other platforms or environments, which can limit the flexibility and interoperability of the program.
- The compiled code may not be secure or verifiable, which can expose the program to potential attacks or errors.

Therefore, JIT compilation is a trade-off between speed and flexibility, and it may not be suitable for every situation or application. It depends on factors like the type of language, the size and complexity of the code, the frequency and duration of execution, and the target platform and environment.

### lax.cond with vmap
When attempting to use [lax.cond](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.cond.html) and wrapping the function with vmap, lax.cond becomes lax.select()

**TODO: find out about monads and monoids. This will help with functional programming design patterns.**