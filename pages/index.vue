<template>
    <div class="flex flex-col items-center justify-center py-15 dark:text-white">
        <h1 class="text-lg font-sans">Image goes here</h1>
        <p>lorem awodhaowuihdowahdaw waod unawoudnawoudnawoud</p>
    </div>
    <div class="flex justify-center items-center mb-20">
        <div class="relative border-l dark:border-gray-400">
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
                    <h3 class="font-bold">{{ job.title }}</h3>
                </div>
                <h4 class="">{{ job.company }}</h4>
                <p class="text-gray-400 text-sm italic">{{ job.date }}</p>
                <p class="">{{ job.description }}</p>
            </div>
        </div>
    </div>
    <p class="text-lg"></p>
</template>

<script lang="ts">
import { timeline } from "~/data/timeline"
export default {
    data() {
        return {
            timelineData: timeline
        };
    },
};
</script>
