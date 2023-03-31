<template>
    <div class="text-white mb-20 text-xl max-w-3xl">
        <h1>
            Welcome to my projects page where I share my passion for coding and creating awesome
            things. Here you can find all the cool projects I have worked on and learn more about
            how I made them. Whether you are a beginner or an expert, I hope you will find something
            interesting and inspiring here.
        </h1>
    </div>

    <div class="grid lg:grid-cols-2 gap-6 grid-cols-1">
        <div v-for="p in data" :key="p.id">
            <projectCard :project="p" />
        </div>
    </div>
</template>

<script setup lang="ts">
const { data } = await useAsyncData("projects", () => queryContent("/projects").find());
</script>

<style scoped></style>
