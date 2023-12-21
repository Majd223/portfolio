<template>
    <div class="flex flex-col items-center justify-center min-h-screen mb-20">
        <div class="relative border-l dark:border-gray-400 w-full max-w-3xl">
            <div
                class="dark:text-white ml-7 last:mb-0 mb-7"
                v-for="(job, index) in timelineData"
                :key="index"
            >
                <div alt="header with image" class="relative">
                    <img
                        alt="bulb"
                        class="absolute -left-10 w-6 h-6 rounded-full ring-8 dark:ring-gray-900 dark:bg-blue-900"
                        :src="job.image"
                    />
                    <h3 class="font-bold">
                        {{ job.title }}
                        <span
                            v-if="job.currently"
                            class="bg-blue-100 text-blue-800 text-sm font-medium mr-2 px-2.5 py-0.5 rounded dark:bg-blue-200 dark:text-blue-800 ml-3"
                            >Currently</span
                        >
                    </h3>
                </div>
                <h4 class="text-sky-300">{{ job.company }}</h4>
                <p class="text-gray-400 text-sm italic">{{ job.date }}</p>
                <p class="max-w-2xl">{{ job.description }}</p>
            </div>
        </div>
        <div
            class="mt-10 text-white w-full max-w-3xl"
            v-for="(section, index) in sectionsData"
            :key="index"
        >
            <h2 class="font-bold text-2xl mb-5">{{ section.title }}</h2>
            <ul class="flex flex-wrap">
                <li
                    v-for="(item, itemIndex) in section.data"
                    :key="itemIndex"
                    class="flex items-center mr-4 mb-4"
                >
                    {{ item }}
                    <span v-if="itemIndex < section.data.length - 1" class="text-lg mx-2">•</span>
                </li>
            </ul>
        </div>
    </div>
</template>

<script setup lang="ts">
import {
    timeline,
    prog_languages,
    libraries,
    tools,
    technologies,
    concepts,
} from "~/data/timeline";
const timelineData = ref(timeline);
const sectionsData = ref([
    { title: "Programming Languages", data: prog_languages },
    { title: "Libraries", data: libraries },
    { title: "Tools", data: tools },
    { title: "Technologies", data: technologies },
    { title: "Concepts", data: concepts },
]);
</script>

<style scoped></style>
