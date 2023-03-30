<template>
    <div class="text-white mb-20">
        <h1>Hellow</h1>
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
