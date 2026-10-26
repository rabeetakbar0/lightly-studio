import type { DimensionBounds } from '$lib/services/loadDimensionBounds';
import { loadDimensionBounds } from '$lib/services/loadDimensionBounds';
import { useSessionStorage } from '$lib/hooks/useSessionStorage/useSessionStorage';
import { isReadableStore } from '$lib/hooks/utils/isReadableStore';
import { derived, get, writable, type Readable } from 'svelte/store';

const dimensionsBounds = useSessionStorage<DimensionBounds | null>(
    'lightlyStudio_dimensions_bounds',
    null
);

const dimensionsValues = useSessionStorage<DimensionBounds | null>(
    'lightlyStudio_dimensions_values',
    null
);

const updateDimensionsValues = (bounds: DimensionBounds) => {
    dimensionsValues.set(bounds);
};

const updateDimensionsBounds = (bounds: DimensionBounds) => {
    dimensionsBounds.set(bounds);
};

const lastCollectionId = writable<string | null>(null);
type CollectionIdInput = string | Readable<string> | undefined;

const loadInitialDimensionBounds = async (collection_id: string) => {
    if (get(lastCollectionId) === collection_id) {
        return;
    }

    lastCollectionId.set(collection_id);
    dimensionsBounds.set(null);
    dimensionsValues.set(null);

    const { data: dimensionBoundsData } = await loadDimensionBounds({
        collection_id: collection_id
    });

    if (get(lastCollectionId) !== collection_id) {
        return;
    }

    // A collection without images has no bounds; the API then returns an empty object.
    if (hasBounds(dimensionBoundsData)) {
        dimensionsBounds.set(dimensionBoundsData);
        dimensionsValues.set(dimensionBoundsData);
    }
};

const hasBounds = (bounds: DimensionBounds | undefined): bounds is DimensionBounds =>
    bounds != null &&
    [bounds.min_width, bounds.max_width, bounds.min_height, bounds.max_height].every(
        (value) => typeof value === 'number' && Number.isFinite(value)
    );

/** Reloads the bounds of a collection, for example after images were added to it. */
export const reloadDimensionBounds = async (collection_id: string) => {
    lastCollectionId.set(null);
    await loadInitialDimensionBounds(collection_id);
};

const bindCollectionLoader = <T>(source: Readable<T>, collectionId: CollectionIdInput) => {
    if (!collectionId) {
        return source;
    }

    if (!isReadableStore<string>(collectionId)) {
        loadInitialDimensionBounds(collectionId);
        return source;
    }

    return derived([source, collectionId], ([$source, $collectionId]) => {
        if ($collectionId) {
            loadInitialDimensionBounds($collectionId);
        }

        return $source;
    });
};

export const useDimensions = (collection_id?: CollectionIdInput) => {
    return {
        dimensionsBounds: bindCollectionLoader(dimensionsBounds, collection_id),
        dimensionsValues: bindCollectionLoader(dimensionsValues, collection_id),
        updateDimensionsValues,
        updateDimensionsBounds
    };
};
