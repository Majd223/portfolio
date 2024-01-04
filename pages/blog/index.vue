<template>
    <div class="text-white mb-20 text-xl max-w-3xl">
        <h1>Welcome to my Blog page, I will be populating this page soon.</h1>
    </div>

    <div class="grid lg:grid-cols-2 gap-6 grid-cols-1">
        <ContentList path="/blog" v-slot="{ list }">
            <div v-for="p in list" :key="p._path">
                <blogCard :post="p" />
            </div>
        </ContentList>
    </div>
</template>

<style scoped></style>
