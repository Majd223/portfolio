<template>
    <NuxtLink to="/your-link">
        <div
            class="rounded border border-gray-300 cursor-pointer transform hover:-translate-y-1 hover:scale-105 transition duration-100 ease-in-out"
        >
            <div class="pr-32 pl-4 py-4 w-full flex flex-col">
                <img
                        alt="Folder icon"
                        class="h-8 w-8 text-gray-400"
                        src="/icon/folder.svg"
                    />
                <div class="font-bold text-xl text-gray-300">{{ project.title }}</div>
                <p class="text-gray-700 text-base mb-4">{{ project.description }}</p>
                <div class="mt-4 flex">
                    <div v-for="tool in project.tools" :key="tool">
                        <span
                        class="inline-block bg-gray-200 rounded-full px-3 py-1 text-sm font-semibold text-gray-700 mr-2"
                        >{{ tool }}</span
                        >
                    </div>
                </div>
                <a
                    aria-label="Open github profile"
                    class="text-sm text-gray-500 hover:text-gray-600 mt-2"
                    target="_blank"
                    rel="nofollow noopener noreferrer"
                    :href="project.github"
                    >
                    <img
                        alt="github"
                        class="transition-transform hover:scale-90 w-8 h-8"
                        src="/icon/github.svg"
                    /> </a
                >
            </div>
        </div>
    </NuxtLink>
</template>

<script setup lang="ts">
    const { project } = defineProps(['project'])
</script>
