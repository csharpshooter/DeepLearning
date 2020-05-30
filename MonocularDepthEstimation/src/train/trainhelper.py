class TrainHelper:

    def cyclical_lr(max_lr_epoch, epochs, min_lr, max_lr):

        # Additional function to see where on the cycle we are
        def calculate_lr_for_epoch(it, max_lr_epoch, epochs, min_lr, max_lr):
            delta = max_lr - min_lr
            delta_one = delta / (max_lr_epoch - 1)
            delta_two = delta / (epochs - max_lr_epoch)

            if it < max_lr_epoch:
                val = min_lr + (delta_one * it)
                return val
            else:
                val = max_lr - ((it - max_lr_epoch + 1) * delta_two)
                return val

        lr_lambda = lambda it: calculate_lr_for_epoch(it, max_lr_epoch, epochs, min_lr, max_lr)
        return lr_lambda
