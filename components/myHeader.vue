<template>
    <div class="shadow-sm bg-slate-950 border-b-2 border-gray-400">
        <nav class="container mx-auto p-4 flex justify-around">
            <NuxtLink to="/" class="font-bold text-gray-100 text-xl pt-1">Majd S</NuxtLink>
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
<style>
.router-link-exact-active {
        color: #a7bebb;
    }
</style>