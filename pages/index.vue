<template>
    <div class="flex flex-col items-center py-15 dark:text-white max-w-3xl">
        <img alt="avatar image" class="pb-5" src="/icon/avatar.svg" />
        <div class="blog-page">
            <h5>Welcome👋 to my portfolio website!</h5>
            <p>{{ welcomeInfo }}</p>
        </div>
    </div>
</template>

<script setup lang="ts">
import { welcome } from "~/data/welcome";
const welcomeInfo = ref(welcome);
</script>
