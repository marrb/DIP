<script setup lang="ts">
    import { faClose } from '@fortawesome/free-solid-svg-icons';

    interface Props {
        size?: "sm" | "md" | "lg" | "xl";
    }

    const props = withDefaults(defineProps<Props>(), {
        size: "md",
    });

    const visible: Ref<boolean> = ref(false);
    const bodyRef: Ref<HTMLElement> = ref(document?.body);
    const isLocked = useScrollLock(bodyRef);

    const show = () => {
        visible.value = true;

        if (bodyRef.value) {
            isLocked.value = true;
        }
    };

    const close = () => {
        visible.value = false;

        if (bodyRef.value) {
            isLocked.value = false;
        }
    };

    const onOverlayClick = (event: MouseEvent) => {
        const selection = window.getSelection()?.toString();
        if (selection && selection.trim() !== "") {
            return;
        }

        close();
    };

    const handleKeydown = (event: any) => {
        if (event.key === "Escape") {
            close();
        }
    };

    onMounted(() => {
        window.addEventListener("keydown", handleKeydown);
    });

    onUnmounted(() => {
        window.removeEventListener("keydown", handleKeydown);
    });

    defineExpose({
        show,
        close,
    });
</script>

<template>
    <div
        v-if="visible"
        @click.self="onOverlayClick"
        class="fixed top-0 bottom-0 left-0 right-0 flex justify-center items-center bg-[rgba(51,51,51,0.8)] z-50">
        <font-awesome
            :icon="faClose"
            class="absolute top-4 right-4 text-white hover:text-blue-400 cursor-pointer text-3xl"
            @click="close" />
        <div
            class="w-full bg-white border-gray-200 border-2 overflow-auto h-auto"
            :class="[
                {
                    'max-w-xl': size === 'sm',
                    'max-w-2xl': size === 'md',
                    'max-w-4xl': size === 'lg',
                    'max-w-6xl': size === 'xl',
                },
            ]">
            <slot></slot>
        </div>
    </div>
</template>
