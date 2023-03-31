<template>
    <div class="text-white mb-20 text-xl max-w-3xl">
        <h1>Welcome to my Blog page, I will be populating this page soon.</h1>
    </div>

    <div class="grid lg:grid-cols-2 gap-6 grid-cols-1">
        <div v-for="p in data" :key="p.id">
            <blogCard :post="p" />
        </div>
    </div>
</template>

<script setup lang="ts">
const { data } = await useAsyncData("blog", () => queryContent("/blog").find());
</script>

<style scoped></style>
