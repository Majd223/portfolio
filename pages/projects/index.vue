<template>
    <div class="mb-20 text-xl max-w-3xl text-white">
        <p>
            Welcome to my projects page where I share my passion for coding and creating awesome
            things. Here you can find all the cool projects I have worked on and learn more about
            how I made them. Whether you are a beginner or an expert, I hope you will find something
            interesting and inspiring here.
        </p>
    </div>

    <div class="grid lg:grid-cols-2 gap-6 grid-cols-1">
        <ContentList path="/projects" v-slot="{ list }">
            <div v-for="p in list" :key="p.id">
                <projectCard :project="p" />
            </div>
        </ContentList>
    </div>
</template>

<style scoped></style>
