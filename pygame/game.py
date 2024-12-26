import pygame
import random
import os

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 400, 600
FPS = 60
PLAYER_SIZE = 70       # Player size
FALLING_OBJECT_SIZE = (80, 100)  # Size for falling objects
BOMB_SIZE = (25, 25)   # Size for bombs
AI_OBJECT_SIZE = (50, 50)  # Size for AI-controlled objects

# Colors
WHITE = (255, 255, 255)

# Load Assets
def load_image(name, size=None):
    image = pygame.image.load(os.path.join('assets', name))
    if size:
        image = pygame.transform.scale(image, size)
    return image

# Load Sounds
def load_sound(name):
    return pygame.mixer.Sound(os.path.join('assets', name))

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Catch the Falling Objects and Avoid Bombs")

# Player Class
class Player:
    def __init__(self):
        self.image = load_image("player.png", (PLAYER_SIZE, PLAYER_SIZE))  # Scale player image
        self.x = WIDTH // 2
        self.y = HEIGHT - PLAYER_SIZE - 10

    def draw(self):
        screen.blit(self.image, (self.x, self.y))

    def move(self, dx):
        self.x += dx
        self.x = max(0, min(self.x, WIDTH - PLAYER_SIZE))  # Keep within bounds

# AI Controlled Falling Object Class
class AIFallingObject:
    def __init__(self, object_type):
        self.type = object_type
        self.image = load_image(f"{object_type}.png", AI_OBJECT_SIZE)  # Load type-specific image
        self.sound = load_sound(f"{object_type}_sound.mp3")  # Load type-specific sound
        self.x = random.randint(0, WIDTH - AI_OBJECT_SIZE[0])
        self.y = 0
        self.speed = 3

    def fall(self, player_x):
        self.y += self.speed  # Fall speed
        # Simple AI behavior to move towards the player
        if self.x < player_x:
            self.x += 1  # Move right
        elif self.x > player_x:
            self.x -= 1  # Move left

    def draw(self):
        screen.blit(self.image, (self.x, self.y))

# Bomb Class
class Bomb:
    def __init__(self):
        self.image = load_image("bomb.png", BOMB_SIZE)  # Scale bomb image
        self.x = random.randint(0, WIDTH - BOMB_SIZE[0])
        self.y = 0

    def fall(self):
        self.y += 5  # Fall speed

    def draw(self):
        screen.blit(self.image, (self.x, self.y))

# Main Game Loop
def main():
    clock = pygame.time.Clock()
    player = Player()
    bombs = []
    ai_objects = []
    score = 0
    running = True
    game_over = False

    # Load Background Music
    pygame.mixer.music.load(os.path.join('assets', "background_music.mp3"))  # Replace with your music
    pygame.mixer.music.play(-1)  # Play background music indefinitely

    # Load Game Over Sound
    game_over_sound = load_sound("game_over.wav")  # Replace with your game over sound file

    while running:
        screen.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            player.move(-5)
        if keys[pygame.K_RIGHT]:
            player.move(5)

        if not game_over:
            # Create new AI-controlled falling objects and bombs
            if random.randint(0, 20) == 0:
                object_type = random.choice(['jethya', 'daya', 'bapuji','babitaji'])  # Randomly choose an object type
                ai_objects.append(AIFallingObject(object_type))
            if random.randint(0, 50) == 0:
                bombs.append(Bomb())

            for ai_object in ai_objects[:]:
                ai_object.fall(player.x)  # AI moves towards player
                ai_object.draw()
                if ai_object.y > HEIGHT:
                    ai_objects.remove(ai_object)  # Remove off-screen objects
                if (player.x < ai_object.x + AI_OBJECT_SIZE[0] and
                    player.x + PLAYER_SIZE > ai_object.x and
                    player.y < ai_object.y + AI_OBJECT_SIZE[1] and
                    player.y + PLAYER_SIZE > ai_object.y):
                    score += 1  # Increment score
                    ai_object.sound.play()  # Play the specific sound for the object
                    ai_objects.remove(ai_object)

            for bomb in bombs[:]:
                bomb.fall()
                bomb.draw()
                if bomb.y > HEIGHT:
                    bombs.remove(bomb)  # Remove off-screen bombs
                if (player.x < bomb.x + BOMB_SIZE[0] and
                    player.x + PLAYER_SIZE > bomb.x and
                    player.y < bomb.y + BOMB_SIZE[1] and
                    player.y + PLAYER_SIZE > bomb.y):
                    pygame.mixer.music.stop()  # Stop music on game over
                    game_over_sound.play()  # Play game over sound
                    game_over = True

            # Draw the player
            player.draw()

            # Display score
            font = pygame.font.Font(None, 36)
            score_text = font.render(f'Score: {score}', True, (0, 0, 0))
            screen.blit(score_text, (10, 10))

        else:
            font = pygame.font.Font(None, 48)
            game_over_text = font.render('Game Over!', True, (255, 0, 0))
            restart_text = font.render('Press R to Restart', True, (0, 0, 0))
            screen.blit(game_over_text, (WIDTH // 2 - 100, HEIGHT // 2 - 50))
            screen.blit(restart_text, (WIDTH // 2 - 150, HEIGHT // 2))

            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                main()  # Restart the game

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
