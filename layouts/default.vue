<template>
    <section class="w-full pb-12 antialiased">
        <div class="mx-auto max-w-8xl">
            <myHeader />
            <div class="flex justify-center items-center flex-col mt-10">
                <slot />
            </div>
            <myFooter />
        </div>
    </section>
</template>

<script></script>

<style scoped>
.router-link-exact-active {
    color: #94a3b8;
}
</style>
