<script lang="ts">
    import { useGlobalStorage, useVideoOrderBy } from '$lib/hooks';
    import { type SelectItem } from '$lib/components/Select';
    import OrderByControl from './OrderByControl.svelte';

    const { textEmbedding } = useGlobalStorage();
    const isSimilaritySearchActive = $derived(!!$textEmbedding);

    const {
        allSortFields,
        selectedDirection,
        selectedLabel,
        isFieldSelected,
        handleFieldClick,
        toggleDirection
    } = useVideoOrderBy();

    const selectValue = $derived.by(() => {
        const idx = $allSortFields.findIndex((field) => $isFieldSelected(field));
        return idx >= 0 ? String(idx) : '';
    });

    const sortItems = $derived<SelectItem[]>(
        $allSortFields.map((field, i) => ({
            value: String(i),
            label: field.label,
            testId: `sort-field-${field.value}`
        }))
    );

    const handleValueChange = (value: string) => {
        if (value === '') {
            const idx = $allSortFields.findIndex((field) => $isFieldSelected(field));
            if (idx >= 0) handleFieldClick($allSortFields[idx]);
            return;
        }
        const field = $allSortFields[Number(value)];
        if (field) handleFieldClick(field);
    };
</script>

<OrderByControl
    items={sortItems}
    selectedValue={selectValue}
    triggerLabel={$selectedLabel ?? undefined}
    direction={$selectedDirection}
    disabled={isSimilaritySearchActive}
    allowDeselect
    onValueChange={handleValueChange}
    onToggleDirection={toggleDirection}
/>
