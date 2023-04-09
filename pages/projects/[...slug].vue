<template>
    <div>
        <main>
            <ContentDoc class="max-w-3xl">
                <template #not-found>
                    <h1 class="text-white">Document not found</h1>
                </template>
                <template #empty>
                    <h1 class="text-white">Document is empty</h1>
                </template>
            </ContentDoc>
        </main>
    </div>
</template>

<style>
main p {
    @apply text-white
}
</style>
