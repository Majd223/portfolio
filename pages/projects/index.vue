<template>
    <div class="flex justify-center items-center mt-10">
        <div class="grid grid-cols-2 gap-4">
            <div v-for="p in projects" :key="p.id">
                <projectCard :project="p" />
            </div>
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
