<template>
    <NuxtLink :to="`/projects/${project.route}`">
        <div
            class="rounded border border-gray-300 cursor-pointer transform hover:-translate-y-1 hover:scale-105 transition duration-100 ease-in-out max-w-xl h-60"
        >
            <div class="pr-10 pl-4 py-4 flex flex-col justify-between">
                <img alt="Folder icon" class="h-8 w-8 text-gray-400 mb-3" src="/icon/folder.svg" />
                <div class="font-bold text-xl text-gray-300">{{ project.title }}</div>
                <p class="text-gray-700 text-base mb-4 line-clamp-2">{{ project.description }}</p>
                <div class="absolute bottom-3">
                    <div class="mt-2 flex mb-3">
                        <div v-for="tool in project.tools" :key="tool">
                            <span
                                class="inline-block bg-gray-200 rounded-full px-3 py-1 text-sm font-semibold text-gray-700 mr-2"
                                >{{ tool }}</span
                            >
                        </div>
                    </div>
                    <a
                        aria-label="Open github profile"
                        class="text-sm text-gray-500 hover:text-gray-600"
                        target="_blank"
                        rel="nofollow noopener noreferrer"
                        :href="project.github"
                    >
                        <img
                            alt="github"
                            class="transition-transform hover:scale-90 w-8 h-8"
                            src="/icon/github.svg"
                        />
                    </a>
                </div>
            </div>
        </div>
    </NuxtLink>
</template>

<script setup lang="ts">
const { project } = defineProps(["project"]);
</script>
