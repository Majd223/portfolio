<template>
    <footer class="lg:mt-40 border-t-2">
        <div class="flex flex-col items-center mt-16">
            <div class="flex mb-3 space-x-4">
                <a
                    aria-label="Open github profile"
                    class="text-sm text-gray-500 transition hover:text-gray-600"
                    target="_blank"
                    rel="nofollow noopener noreferrer"
                    :href="siteMetadata.github"
                    ><span class="sr-only">github</span>
                    <img
                        alt="github"
                        class="transition-transform hover:scale-110 w-8 h-8"
                        src="/icon/github.svg"
                    /> </a
                ><a
                    aria-label="Open linkedin profile"
                    class="text-sm text-gray-500 transition hover:text-gray-600"
                    target="_blank"
                    rel="noopener noreferrer"
                    :href="siteMetadata.linkedin"
                    ><span class="sr-only">Linkedin</span>
                    <img
                        alt="Linkedin"
                        class="transition-transform hover:scale-110 w-8 h-8"
                        src="/icon/linkedin.svg"
                    />
                </a>
                <a
                    aria-label="Open twitter profile"
                    target="_blank"
                    rel="nofollow noopener noreferrer"
                    :href="siteMetadata.twitter"
                    class="text-sm text-gray-500 transition hover:text-gray-600"
                >
                    <img
                        alt="Twitter"
                        class="transition-transform hover:scale-110 w-8 h-8"
                        src="/icon/twitter.svg"
                    />
                </a>
            </div>
            <div class="flex mb-2 space-x-2 text-sm text-gray-500 dark:text-gray-400">
                <div>Copyright © {{ new Date().getFullYear() }}</div>
                <div>• Majd Safi</div>
            </div>
        </div>
    </footer>
</template>

<script lang="ts">
import { siteMetaData } from "~/data/siteMetaData";
export default {
    data() {
        return {
            siteMetadata: ref(siteMetaData),
        };
    },
};
</script>
