
# First Run

alpha01 <- read.table("hyperparams_alpha01_gamma09.txt", header = T, sep = "")
alpha03 <- read.table("hyperparams_alpha03_gamma07.txt", header = T, sep = "")
alpha05 <- read.table("hyperparams_alpha05_gamma05.txt", header = T, sep = "")
alpha07 <- read.table("hyperparams_alpha07_gamma03.txt", header = T, sep = "")
alpha09 <- read.table("hyperparams_alpha09_gamma01.txt", header = T, sep = "")



alpha01$Setting <- "alpha=0.1, gamma=0.9"
alpha03$Setting <- "alpha=0.3, gamma=0.7"
alpha05$Setting <- "alpha=0.5, gamma=0.5"
alpha07$Setting <- "alpha=0.7, gamma=0.3"
alpha09$Setting <- "alpha=0.9, gamma=0.1"


data <- rbind(alpha01, alpha03, alpha05, alpha07, alpha09)

ggplot(data, aes(x = Episode, y = TotalReward, color = Setting)) +
  geom_line(alpha = 0.2, linewidth = 0.6) +   
  geom_smooth(se = FALSE, span = 0.2, linewidth = 1.2) +  
  labs(title = "Q-Learning Convergence by Hyperparameter Setting",
       subtitle = "Smoothing applied to clarify trends",
       x = "Episode", y = "Total Reward", color = "Hyperparameter Setting") +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "right",               
    legend.key.width = unit(1.5, "cm"),      
    legend.text = element_text(size = 10),   
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 12)
  ) +
  guides(color = guide_legend(ncol = 1))     


library(dplyr)
# Obtaining the last 100 iteration's average.
summary <- data %>%
  group_by(Setting) %>%
  filter(Episode > max(Episode) - 100) %>%
  summarise(AvgFinalReward = mean(TotalReward),
            MaxReward = max(TotalReward),
            MinReward = min(TotalReward))

print(summary)

top10_rewards <- alpha03 %>%
  arrange(desc(TotalReward)) %>%
  head(10)

print(top10_rewards)


# Second run
alpha02 <- read.table("hyperparams_alpha02_gamma08.txt", header = T, sep = "")
alpha03 <- read.table("hyperparams_alpha03_gamma07.txt", header = T, sep = "")
alpha04 <- read.table("hyperparams_alpha04_gamma09.txt", header = T, sep = "")
alpha038 <- read.table("hyperparams_alpha03_gamma08.txt", header = T, sep = "")
alpha039 <- read.table("hyperparams_alpha03_gamma09.txt", header = T, sep = "")



alpha02$Setting <- "alpha=0.2, gamma=0.8"
alpha03$Setting <- "alpha=0.3, gamma=0.7"
alpha04$Setting <- "alpha=0.4, gamma=0.9"
alpha038$Setting <- "alpha=0.3, gamma=0.8"
alpha039$Setting <- "alpha=0.3, gamma=0.9"

data2 <- rbind(alpha02, alpha03, alpha04, alpha038, alpha039)


ggplot(data2, aes(x = Episode, y = TotalReward, color = Setting)) +
  geom_line(alpha = 0.2, linewidth = 0.6) +   
  geom_smooth(se = FALSE, span = 0.2, linewidth = 1.2) +  
  labs(title = "Q-Learning Convergence by Hyperparameter Setting",
       subtitle = "Smoothing applied to clarify trends",
       x = "Episode", y = "Total Reward", color = "Hyperparameter Setting") +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "right",               
    legend.key.width = unit(1.5, "cm"),      
    legend.text = element_text(size = 10),   
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 12)
  ) +
  guides(color = guide_legend(ncol = 1))     


library(dplyr)
# Obtaining the last 100 iteration's average.
summary <- data2 %>%
  group_by(Setting) %>%
  filter(Episode > max(Episode) - 100) %>%
  summarise(AvgFinalReward = mean(TotalReward),
            MaxReward = max(TotalReward),
            MinReward = min(TotalReward))

print(summary)

top10_rewards <- alpha03 %>%
  arrange(desc(TotalReward)) %>%
  head(10)

print(top10_rewards)