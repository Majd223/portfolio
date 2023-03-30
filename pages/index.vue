<template>
    <div class="flex flex-col items-center py-15 dark:text-white">
        <img alt="avatar image" class="pb-5" src="../public/icon/avatar.svg"/>
        <h1 class="pb-16 text-xl">Hi! I am Majd and this is my website</h1>
    </div>
    <div class="mb-20">
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
                    <h3 class="font-bold">
                        {{ job.title }}
                        <span
                            v-if="job.currently"
                            class="bg-blue-100 text-blue-800 text-sm font-medium mr-2 px-2.5 py-0.5 rounded dark:bg-blue-200 dark:text-blue-800 ml-3"
                            >Currently</span
                        >
                    </h3>
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
import { timeline } from "~/data/timeline";
export default {
    data() {
        return {
            timelineData: ref(timeline),
        };
    },
};
</script>
