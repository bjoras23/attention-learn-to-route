def log_values(cost, pen, reward, bl_reward, grad_norms, epoch, batch_id, step,
               log_likelihood, reinforce_loss, bl_loss, tb_logger, opts):
    avg_cost = cost.mean().item()
    avg_pen = pen.mean().item()
    avg_r = reward.mean().item()
    avg_bl_r = bl_reward.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print(f'epoch: {epoch}, train_batch_id: {batch_id}, avg_cost: {avg_cost:.5}, avg_pen: {avg_pen:.5}, avg_reward: {avg_r:.5}')

    print(f'grad_norm: {grad_norms[0]:.5}, clipped: {grad_norms_clipped[0]}')

    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.add_scalar('train/avg_cost', avg_cost, step)
        tb_logger.add_scalar('train/avg_penalty', avg_pen, step)
        tb_logger.add_scalar('train/avg_reward', avg_r, step)
        tb_logger.add_scalar('train/avg_baseline_reward', avg_bl_r, step)

        tb_logger.add_scalar('train/actor_loss', reinforce_loss.item(), step)
        tb_logger.add_scalar('train/nll', -log_likelihood.mean().item(), step)

        tb_logger.add_scalar('train/grad_norm', grad_norms[0], step)
        tb_logger.add_scalar('train/grad_norm_clipped', grad_norms_clipped[0], step)

        if opts.baseline == 'critic':
            tb_logger.add_scalar('train/critic_loss', bl_loss.item(), step)
            tb_logger.add_scalar('train/critic_grad_norm', grad_norms[1], step)
            tb_logger.add_scalar('train/critic_grad_norm_clipped', grad_norms_clipped[1], step)
