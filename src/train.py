from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score


class Trainer:
    def __init__(self,
                 model, 
                 criterion, 
                 optimizer, 
                 scheduler,
                 train_loader, 
                 val_loader, 
                 test_loader, 
                 device, 
                 experiment_id, 
                 base_dir
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.save_model_dir = base_dir / "models" / experiment_id
        self.save_model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.save_model_dir / "best_model.pth"

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_val_acc = 0

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        for X, y in self.train_loader:
            X, y = X.to(self.device), y.to(self.device).unsqueeze(1).float()
            y_pred = self.model(X)
            loss = self.criterion(y_pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device).unsqueeze(1).float()
                y_pred = self.model(X)

                probs = torch.sigmoid(y_pred)
                preds = (probs > 0.5).int()

                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(y.cpu().numpy().flatten())
        return accuracy_score(all_labels, all_preds)

    def train(self, num_epochs):
        print("Training the model...")
        for epoch in tqdm(range(num_epochs)):
            loss = self.train_one_epoch()
            val_acc = self.evaluate()
            self.scheduler.step(val_acc)

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Validation Accuracy: {val_acc:.4f}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc

                torch.save(self.model.state_dict(), self.model_path)
                print("Model saved.")

    def test(self):
        print("Testing the model...")
        self.model.load_state_dict(torch.load(self.model_path))
        acc = self.evaluate(self.test_loader)
        print(f"Test Accuracy: {acc:.4f}")
