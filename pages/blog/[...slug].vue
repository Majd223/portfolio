<template>
    <div class="blog-page">
        <main>
            <ContentDoc class="max-w-3xl">
                <template #not-found>
                    <h1>Document not found</h1>
                </template>
                <template #empty>
                    <h1>Document is empty</h1>
                </template>
            </ContentDoc>
        </main>
    </div>
</template>

<script lang="ts"></script>

<style>
.blog-page h1,
.blog-page h2,
.blog-page h3,
.blog-page h4,
.blog-page h5,
.blog-page p,
.blog-page pre {
    @apply text-white;
    margin-bottom: 1em;
    @apply font-sans antialiased;
}

.blog-page ul {
    @apply text-slate-400;
    list-style-type: disc;
    margin-bottom: 1em;
}

.blog-page p {
    @apply text-white text-justify;
    text-indent: 2em;
    line-height: 1.6;
}

.blog-page h1 {
    @apply text-center;
    font-size: 2em;
}

.blog-page pre {
    @apply overflow-auto;
}

.blog-page h2 {
    font-size: 1.5em;
}

.blog-page p {
    font-size: 1em;
}

.blog-page a {
    color: #0099cc;
    text-decoration: none;
}

.blog-page a:hover {
    text-decoration: underline;
}
</style>
