<template>
    <div class="flex flex-col items-center py-15 dark:text-white max-w-3xl">
        <img alt="avatar image" class="pb-5" src="../icon/avatar.svg" />
        <div class="pb-16 text-xl">
            <h1>Welcome👋 to my portfolio webiste!</h1>
            <h1>{{ welcomeInfo }}</h1>
        </div>
    </div>
</template>

<script setup lang="ts">
import { welcome } from "~/data/welcome";
const welcomeInfo = ref(welcome);
</script>
