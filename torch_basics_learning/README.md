# PyTorch Basic Features Documentation

This documentation showcases 62 fundamental features of PyTorch, offering hands-on examples for everything from tensor creation to advanced operations like Einstein summation and sparse tensors. It serves as a comprehensive guide for beginners and intermediate users to understand and implement essential PyTorch capabilities.

---

## 1. Tensor Creation

- **From Python lists:**
  ```python
  tensor_from_list = torch.tensor([1, 2, 3])
  ```
- **Filled with zeros/ones:**
  ```python
  zeros_tensor = torch.zeros((2, 3))
  ones_tensor = torch.ones((2, 3))
  ```
- **Random tensors:**
  ```python
  random_uniform = torch.rand(2, 3)
  random_normal = torch.randn(2, 3)
  ```
- **Range of values:**
  ```python
  range_tensor = torch.arange(1, 10, 2)
  ```
- **Filled with a fixed value:**
  ```python
  full_tensor = torch.full((2, 3), fill_value=5)
  ```

## 2. Tensor Operations

- **Arithmetic operations:**
  ```python
  addition = x + y
  multiplication = x * y
  ```
- **Matrix multiplication:**
  ```python
  matmul_result = torch.matmul(matrix1, matrix2)
  ```
- **Reshaping tensors:**
  ```python
  reshaped_tensor = tensor.view(2, 3)
  ```
- **Concatenation and stacking:**
  ```python
  concatenated = torch.cat((t1, t2), dim=0)
  stacked = torch.stack((t1, t2), dim=1)
  ```
- **Broadcasting:**
  ```python
  broadcasted = b + c
  ```
- **Element-wise math functions:**
  ```python
  exp_tensor = torch.exp(random_uniform)
  log_tensor = torch.log(random_uniform)
  sqrt_tensor = torch.sqrt(random_uniform)
  ```
- **Reduction operations:**
  ```python
  sum_result = random_uniform.sum()
  mean_result = random_uniform.mean()
  max_result = random_uniform.max()
  ```

## 3. Indexing and Slicing

- **Basic indexing:**
  ```python
  first_row = tensor[0]
  ```
- **Advanced indexing:**
  ```python
  selected_rows = torch.index_select(tensor, dim=0, index=indices)
  gathered = torch.gather(tensor, dim=1, index=indices)
  ```

## 4. Boolean Masking and Comparison

- **Comparison operators:**
  ```python
  comparison = random_uniform > 0.5
  ```
- **Boolean masking:**
  ```python
  masked_values = random_uniform[comparison]
  ```
- **Logical operations:**
  ```python
  logical_and = torch.logical_and(tensor1, tensor2)
  logical_or = torch.logical_or(tensor1, tensor2)
  logical_not = torch.logical_not(tensor1)
  ```

## 5. Sorting and Top-k Elements

- **Sorting:**
  ```python
  sorted_tensor, indices = torch.sort(random_uniform, dim=1, descending=True)
  ```
- **Top-k:**
  ```python
  topk_values, topk_indices = torch.topk(random_uniform, k=2, dim=1)
  ```

## 6. Padding and Splitting

- **Padding:**
  ```python
  padded_tensor = torch.nn.functional.pad(tensor, pad=(1, 1, 1, 1), mode='constant', value=0)
  ```
- **Splitting:**
  ```python
  split_tensors = torch.split(tensor, split_size_or_sections=1, dim=0)
  ```

## 7. Special Functions and Transformations

- **Trigonometric functions:**
  ```python
  sin_tensor = torch.sin(random_uniform)
  cos_tensor = torch.cos(random_uniform)
  tanh_tensor = torch.tanh(random_uniform)
  ```
- **Permuting and transposing:**
  ```python
  permuted_tensor = tensor.permute(1, 0)
  transposed_tensor = tensor.transpose(0, 1)
  ```
- **Clamping:**
  ```python
  clamped_tensor = random_uniform.clamp(min=0.2, max=0.8)
  ```

## 8. Type Conversion and Device Management

- **Type conversion:**
  ```python
  int_tensor = float_tensor.to(torch.int32)
  ```
- **Move to GPU:**
  ```python
  tensor_on_gpu = random_uniform.to(device)
  ```

## 9. Miscellaneous Features

- **Cumulative sums/products:**
  ```python
  cumsum_tensor = torch.cumsum(random_uniform, dim=1)
  cumprod_tensor = torch.cumprod(random_uniform, dim=1)
  ```
- **Unique values and counts:**
  ```python
  unique_values, counts = torch.unique(tensor, return_counts=True)
  ```
- **Outer product:**
  ```python
  outer_product = torch.outer(x, y)
  ```
- **Argmax/Argmin:**
  ```python
  argmax_value = torch.argmax(tensor)
  argmin_value = torch.argmin(tensor)
  ```
- **Rolling tensors:**
  ```python
  rolled_tensor = torch.roll(tensor, shifts=1, dims=0)
  ```
- **Sparse tensors:**
  ```python
  sparse_tensor = torch.sparse_coo_tensor(indices, values, size=(2, 3))
  dense_tensor = sparse_tensor.to_dense()
  ```
- **Broadcasting examples:**
  ```python
  broadcasted_result = broadcast_tensor + torch.tensor([10, 20, 30])
  ```
- **Normalization:**
  ```python
  normalized_tensor = torch.nn.functional.normalize(random_uniform, p=2, dim=1)
  ```
- **One-hot encoding:**
  ```python
  one_hot = torch.nn.functional.one_hot(labels, num_classes=3)
  ```
- **Handling NaNs:**
  ```python
  clean_tensor = nan_tensor[~torch.isnan(nan_tensor)]
  ```
- **Chunking:**
  ```python
  chunked_tensors = torch.chunk(tensor, chunks=3, dim=0)
  ```
- **Flattening:**
  ```python
  flattened_tensor = tensor.flatten()
  ```
- **Repeating and tiling:**
  ```python
  repeated_tensor = tensor.repeat(2, 3)
  expanded_tensor = tensor.expand(2, 3)
  ```
- **Diagonal extraction/embedding:**
  ```python
  diagonal = torch.diagonal(tensor)
  diag_matrix = torch.diag_embed(tensor)
  ```
- **Conditional operations:**
  ```python
  where_tensor = torch.where(condition, tensor, torch.zeros_like(tensor))
  ```
- **Histogram computation:**
  ```python
  histogram = torch.histc(random_uniform, bins=5, min=0, max=1)
  ```
- **Coordinate grids:**
  ```python
  grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
  ```
- **Einstein summation:**
  ```python
  einsum_result = torch.einsum('ij,jk->ik', a, b)
  ```
- **Scattering values:**
  ```python
  scattered = scatter_tensor.scatter(dim=1, index=indices, src=values)
  ```
- **Mask-based replacement:**
  ```python
  filled_tensor = masked_fill_tensor.masked_fill(mask, value=-1)
  ```

## Conclusion

This program demonstrates 62 basic features of PyTorch, covering everything from tensor creation to advanced operations like Einstein summation and sparse tensors. It provides a step-by-step guide to help you build a strong foundation in PyTorch and encourages further exploration into deep learning applications!

