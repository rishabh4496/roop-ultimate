import { getJSON, postJSON } from './apiClient';

export const storageAdapter = {
  getReview: () => getJSON('/api/storage'),
  deleteItem: (itemId) => postJSON('/api/storage/delete', { item_id: itemId, confirm: true }),
};
