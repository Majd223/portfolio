<template>
    <NuxtLink :to="`${post._path}`">
        <div
            class="rounded card-color cursor-pointer transform hover:-translate-y-1 hover:scale-105 transition duration-100 ease-in-out max-w-xl h-full sm:max-w-sm"
        >
            <div class="px-5 py-4 flex flex-col justify-between">
                <img
                    alt="Blog card image"
                    class="w-xl text-gray-400 mb-3 border border-slate-950"
                    src="/icon/folder.svg"
                />
                <h1 class="font-bold text-xl text-gray-300">{{ post.title }}</h1>
                <div class="bg-indigo-500 py-1 px-2 rounded-md w-fit text-sm">
                    <span alt="date" class="text-white inline">{{ post.date }}</span>
                </div>
                <p class="text-gray-700 text-base mb-4 line-clamp-2">{{ post.description }}</p>
                <div class="absolute bottom-3">
                    <div class="mt-2 flex mb-3"></div>
                </div>
            </div>
        </div>
    </NuxtLink>
</template>

<script setup lang="ts">
const { post } = defineProps(["post"]);
</script>

<style scoped>
.card-color {
    background-color: #0b1c3b;
}
</style>
