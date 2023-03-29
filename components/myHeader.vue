<template>
    <div class="shadow-sm bg-black border-b-2 border-white">
        <nav class="container mx-auto p-4 flex justify-around">
            <NuxtLink to="/" class="font-bold text-white">Majd S</NuxtLink>
            <div class="flex gap-4">
                <NuxtLink v-for="item in navLinks" :key="item.path" :to="item.path" class="btn"
                    >{{ item.name }}
                </NuxtLink>
            </div>
        </nav>
    </div>
</template>

<script setup lang="ts">
import { menu } from "../data/menu";

const navLinks = ref(menu);
</script>
