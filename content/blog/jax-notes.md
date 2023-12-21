---
title: "Notes on JAX" 
route: "jax-notes"
description: "List of notes I acquired from using JAX"
date: "12-20-2023"
---

# JAX
If you are looking for a Python library that can help you write fast and efficient code for different hardware platforms, you might want to check out JAX. JAX is a library that combines three powerful features: just in time compilation, Autograd and XLA.

Just in time compilation (JIT) means that JAX can compile your Python code on the fly and optimize it for the target device. This way, you don't have to write separate code for CPU, GPU, or TPU. You can use the same code and let JAX handle the compilation.

Autograd is a feature that allows you to automatically compute gradients of your functions. This is very useful for machine learning and optimization tasks, where you often need to calculate the derivatives of your loss function with respect to your parameters. With Autograd, you can write your functions in a natural way and let JAX take care of the differentiation.

XLA stands for Accelerated Linear Algebra, and it is a compiler that can generate highly optimized code for linear algebra operations. XLA can leverage the parallelism and memory hierarchy of modern hardware devices, such as GPUs and TPUs, and make your code run faster and more efficiently.

JAX is a library that can help you write code that can be compiled to run on a CPU, GPU, or TPU without changing the code. JAX stands for just-in-time compilation, Autograd, and XLA. XLA is super quick. If you want to learn more about JAX and how to use it, you can visit its official website or check out some tutorials and examples online. In my experience, the best way to learn JAX is through reading books about it, looking through the source code, and of course, reading the documentation.

## JAX notes
### JAX mutable states
Jax HATES mutable state, and thats how functional programming usually works, you try your best to not mutate state. try to keep everything stateless.
```python
import Jax as jnp
X = jnp.array([1, 2, 3, 4, 5])

# You can't do X[3] = 10, JAX will complain
# Instead you have to create a whole new array and assign 
# it and this can be done as the following:
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

##### JIT compilation is not a silver bullet, though. It also has some drawbacks, such as:

- The compilation process itself takes time and resources, which can cause delays or overhead during the execution of the program. This is especially noticeable for short-running programs or scripts that do not benefit much from the optimizations.
- The compiled code may not be portable or compatible with other platforms or environments, which can limit the flexibility and interoperability of the program.
- The compiled code may not be secure or verifiable, which can expose the program to potential attacks or errors.

Therefore, JIT compilation is a trade-off between speed and flexibility, and it may not be suitable for every situation or application. It depends on factors like the type of language, the size and complexity of the code, the frequency and duration of execution, and the target platform and environment.

### lax.cond with vmap
When attempting to use [lax.cond](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.cond.html) and wrapping the function with vmap, lax.cond becomes lax.select()

### Calling matmul
in Python 3.9 the operator @ was introduced to call matmul. Which looks way better syntax-wise than calling the function. So something like this

```python
import numpy as np
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
arr1 @ arr2
```
But when you call this function its unclear if its calling the numpy function or the JAX function. But my guess is that its calling the numpy function. A simple speed test can confirm this.

```python
@jax.jit
def new_matmul(w, c):
    return jnp.matmul(w, c)

@jax.jit
def old_matmul(w, c):
    return w @ c

# where w is an array of floats and c is an array of 0s or 1s
c = generate_c(new_key(), (9_000_000, 128)) # c is 9 million rows by 128 columns
w = generate_w(new_key(), (1, 128)) # one row by 128 columns

%timeit new_matmul(w, c)
%timeit old_matmul(w, c)
```
The results are as follows:
```
337 ms ± 37.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
166 ms ± 8.64 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```
This is enough to show that calling the jnp function is faster than calling the @ operator. So its a good practice to write out the entire function call. Another thing to note is that when calling @ I was not able to fit a big array in my VRAM. But when I started calling the jnp function it worked like a charm! jnp.matmul calls XLA's dot_general under the hood. But its not worth it in my opinion to call dot general directly since jnp.matmul figures out the variables to pass to dot_general for you.

### in-place update under JIT
As we know, JAX doesn't like mutating variables and will create a new array when trying to update a variables. But outside of of JIT, the updates will lead to array copies (because JAX arrays are immutable at the python level). Within JIT, the compiler will apply updates in-place when possible. According to this [array-updates-x-at-idx-set-y](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#array-updates-x-at-idx-set-y) example, it says inside jit-compiled code, if the input value is not reused the compiler will optimize the array update to occue in-place. Pretty neat! I came across this when I was trying to create a function that takes three matrixes and preform interleave concat in jax. 
```python
@jax.jit
def interleave_concat(t1, t2, t3):
    res = jnp.zeros((t1.shape[0] * 3,t1.shape[1]), dtype=jnp.int8)
    return res.at[0::3,:].set(t1).at[1::3,:].set(t2).at[2::3,:].set(t3)
```
This function will take three matrixes and interleave them together. So its t1[0\] t2[0\] t3[0\] t1[1] t2[1\] t3[1\] and so on.