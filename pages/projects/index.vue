<template>
        <div class="text-white mb-20">
            <h1>Hellow</h1>
        </div>
        
        <div class="grid lg:grid-cols-2 gap-6 grid-cols-1">
            <div v-for="p in projects" :key="p.id">
                <projectCard :project="p" />
            </div>
        </div>
</template>

<script lang="ts">
import { projectsData } from "~/data/projectsData";

export default {
    data() {
        return {
            projects: ref(projectsData),
        };
    },
};
</script>

<style scoped></style>
