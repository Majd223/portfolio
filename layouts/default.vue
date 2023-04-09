<template>
    <section class="w-full pb-12 antialiased">
        <div class="mx-auto max-w-8xl">
            <myHeader />
            <div class="flex justify-center items-center flex-col mt-10 pb-16 text-xl">
                <slot />
            </div>
            <myFooter />
        </div>
    </section>
</template>
