<template>
    <div>
        <main>
            <ContentDoc>
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
