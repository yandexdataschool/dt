module dtrain
implicit none
contains

subroutine build_decision_fortran(X, targets, weights, bootstrap_weights, current_indices, columns_to_test, n_thresh, &
    depth, reg, use_friedman_mse, n_threads, feature, cut, best_improvements, best_cuts)
    ! using openmp
    use omp_lib
    ! input
    integer*1, intent(in) :: X(:, :)
    real*4,    intent(in) :: targets(:), weights(:), bootstrap_weights(:)
    integer*4, intent(inout) :: current_indices(:)
    integer*4, intent(in) :: columns_to_test(:)
    integer*4, intent(in) :: n_thresh, depth, use_friedman_mse, n_threads
    real*8,    intent(in) :: reg
    ! output
    integer*4, intent(out) :: feature, cut
    real*8,    intent(out) :: best_improvements(size(columns_to_test))
    integer*4, intent(out) :: best_cuts(size(columns_to_test))

    real*4, allocatable:: bin_gradients_flat(:), bin_hessians_flat(:)
    real*8, allocatable:: bin_gradients(:, :), bin_hessians(:, :)

    real*8, dimension(2 ** (depth - 1)) :: temp, temp_gradients_op, temp_hessians_op
    real*4 :: gradients(size(targets)), hessians(size(targets))
    real*4 :: improvements(0:n_thresh - 1)
    integer*4 :: leaf_indices(size(current_indices)), leaf, n_leaves, thresh, column_id, column, i
    integer*1 :: temp_complementary
    call omp_set_num_threads(n_threads)

    n_leaves = 2 ** (depth - 1)

    !$OMP PARALLEL DO SCHEDULE(STATIC)
    do i = lbound(targets, 1), ubound(targets, 1)
        hessians(i) = weights(i) * bootstrap_weights(i)
        gradients(i) = targets(i) * hessians(i)
        leaf_indices(i) = iand(current_indices(i), n_leaves - 1)
    end do
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO private(bin_gradients_flat, bin_hessians_flat, bin_gradients, bin_hessians, thresh, &
    !$OMP &                   i, temp, column, improvements, leaf, temp_gradients_op, temp_hessians_op) &
    !$OMP & SCHEDULE(DYNAMIC, 1)
    do column_id = lbound(columns_to_test, 1), ubound(columns_to_test, 1)
        column = columns_to_test(column_id) + 1

        allocate(bin_gradients_flat(0:n_thresh * n_leaves - 1))
        allocate(bin_hessians_flat(0:n_thresh * n_leaves - 1))
        allocate(bin_gradients(0:n_leaves - 1, 0:n_thresh - 1))
        allocate(bin_hessians(0:n_leaves - 1, 0:n_thresh - 1))

        bin_gradients_flat(:) = 0
        bin_hessians_flat(:) = 0

        !$OMP SIMD
        do i = 1, size(leaf_indices, 1)
            leaf = IOR(leaf_indices(i), lshift(int(X(i, column), 4), depth - 1))
            bin_gradients_flat(leaf) = bin_gradients_flat(leaf) + gradients(i)
            bin_hessians_flat(leaf)  = bin_hessians_flat(leaf) + weights(i)
        end do

        bin_gradients(:, :) = reshape(bin_gradients_flat, shape(bin_gradients))
        bin_hessians(:, :) = reshape(bin_hessians_flat, shape(bin_hessians))

        do thresh = 1, n_thresh - 1
            bin_gradients(:, thresh) = bin_gradients(:, thresh) + bin_gradients(:, thresh - 1)
            bin_hessians(:, thresh)  = bin_hessians(:, thresh) + bin_hessians(:, thresh - 1)
        end do

        do thresh = 0, n_thresh - 1
            temp_gradients_op(:) = bin_gradients(:, n_thresh - 1) - bin_gradients(:, thresh)
            temp_hessians_op(:) = bin_hessians(:, n_thresh - 1) - bin_hessians(:, thresh)

            if (use_friedman_mse == 0) then
                temp(:) = bin_gradients(:, thresh) ** 2 / (bin_hessians(:, thresh) + reg) + &
                    temp_gradients_op(:) ** 2 / (temp_hessians_op(:) + reg)
            else
                temp(:) = (bin_gradients(:, thresh) * temp_hessians_op(:) - temp_gradients_op(:) * bin_hessians(:, thresh)) ** 2.
                temp(:) = temp(:) / &
                    ((bin_hessians(:, thresh) + reg) * (temp_hessians_op(:) + reg) * (bin_hessians(:, n_thresh - 1) + reg))
            end if
            improvements(thresh) = sum(temp(:))
        end do

        best_improvements(column_id) = maxval(improvements)
        best_cuts(column_id) = maxloc(improvements, 1) - 1
        thresh = best_cuts(column_id)

        deallocate(bin_gradients_flat, bin_hessians_flat, bin_gradients, bin_hessians)
    end do
    !$OMP END PARALLEL DO
    feature = columns_to_test(maxloc(best_improvements, 1))
    cut = best_cuts(maxloc(best_improvements, 1))

    temp_complementary = 128 - 1 - cut
    !$OMP PARALLEL DO SCHEDULE(STATIC)
    do i = lbound(current_indices, 1), ubound(current_indices, 1)
        current_indices(i) = IOR(lshift(current_indices(i), 1), IBITS(X(i, feature + 1) + temp_complementary, 7, 1))
    end do
    !$OMP END PARALLEL DO
end subroutine


subroutine predict_several_trees_fortran(X_64, indices, current_predictions, &
    depth, features, thresholds, leaf_values_array)

    integer*8, intent(in) :: X_64(:, :)
    integer*1, intent(in) :: thresholds(:)
    integer*4, intent(in) :: depth, features(:)
    real*4,    intent(in) :: leaf_values_array(:, :)
    integer*4, intent(inout) :: indices(:)
    real*4,    intent(inout) :: current_predictions(:)

    integer*4 :: i, j, n_c_leaf_values, mask, n_features, shift
    integer*8 :: temp_x64, multiplier_x64, mask_x64
    real*4,    allocatable :: hybrid_leaf_values(:)
    integer*8, allocatable :: computed_indices_x64(:)
    integer*1, allocatable :: computed_indices_x8(:)

    n_c_leaf_values = 2 ** (depth + size(features, 1) - 1)

    allocate(hybrid_leaf_values(0:n_c_leaf_values - 1), &
             computed_indices_x64(size(X_64, 1)), &
             computed_indices_x8(size(X_64, 1) * 8))


    hybrid_leaf_values(:) = 0
    mask = 2 ** depth - 1
    n_features = size(features, 1)
    do i = 0, n_features - 1
        do j = 0, n_c_leaf_values - 1
            hybrid_leaf_values(j) = hybrid_leaf_values(j) + &
                leaf_values_array(n_features - i, 1 + iand(mask, rshift(j, i)))
        end do
    end do

    multiplier_x64 = 1
    do i = 0, 7
        multiplier_x64 = or(multiplier_x64, lshift(multiplier_x64, 8))
    end do
    mask_x64 = lshift(multiplier_x64, 7)

    computed_indices_x64(:) = 0
    do i = 1, n_features
        temp_x64 = (2 ** 7 - 1 - thresholds(i)) * multiplier_x64
        shift = 7 + i - n_features
        computed_indices_x64(:) = ior(computed_indices_x64(:), shiftr(iand(temp_x64 + X_64(:, features(i) + 1), mask_x64), shift))
    end do

    computed_indices_x8(:) = transfer(computed_indices_x64(:), computed_indices_x8(:))
    do i = 1, size(indices, 1)
        indices(i) = ior( lshift(indices(i), n_features), and(computed_indices_x8(i), 255))
        current_predictions(i) = current_predictions(i) + hybrid_leaf_values( iand( indices(i), n_c_leaf_values - 1) )
    end do

    deallocate(hybrid_leaf_values, computed_indices_x64, computed_indices_x8)
end subroutine

subroutine parallel_bincount_fortran(indices, grads, hesss, n_bins, bin_grads, bin_hesss)
    implicit none
    real*4,    intent(in) :: grads(:), hesss(:)
    integer*4, intent(in) :: indices(:), n_bins
    real*8,    intent(out) :: bin_grads(0:n_bins-1), bin_hesss(0:n_bins-1)
    integer :: i, index
    bin_grads(:) = 0
    bin_hesss(:) = 0
    do i = lbound(indices, 1), ubound(indices, 1)
        index = indices(i)
        bin_grads(index) = bin_grads(index) + grads(i)
        bin_hesss(index) = bin_hesss(index) + hesss(i)
    end do
end subroutine

subroutine log_loss_grad(y_signed, pred, n_threads, result)
    real*4,    intent(in) :: y_signed(:), pred(:)
    integer*4, intent(in) :: n_threads
    real*4,    intent(out) :: result(size(pred))
    integer*4 :: i
    !$OMP PARALLEL DO SCHEDULE(STATIC) NUM_THREADS(n_threads)
    do i = lbound(pred, 1), ubound(pred, 1)
        result(i) = y_signed(i) - tanh(pred(i) / 2)
    end do
end subroutine

end module