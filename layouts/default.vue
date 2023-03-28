<template>
    <section class="w-full pb-12 antialiased">
        <div class="mx-auto max-w-8xl">
            <myHeader />
            <slot />
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
