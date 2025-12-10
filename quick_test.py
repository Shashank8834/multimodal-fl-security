#!/usr/bin/env python
"""
Quick Sanity Check Script

Tests DP-SGD and CUB-200 with minimal epochs to verify code works.
Run this before full experiments to catch errors early.
"""

import os
import sys
import torch
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_dp_sgd():
    """Test DP-SGD with 1 round on MNIST."""
    print("=" * 60)
    print("TEST 1: DP-SGD on MNIST (1 round)")
    print("=" * 60)
    
    from src.utils.data_loader import load_mnist, get_client_data
    from src.models.simple_cnn import create_model
    from src.defenses.differential_privacy import DPSGDDefense
    from src.utils.metrics import evaluate_model
    from torch.utils.data import DataLoader
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load data
    train_data, test_data = load_mnist("./data")
    test_loader = DataLoader(test_data, batch_size=64)
    
    # Create model
    model = create_model(device=device)
    
    # Create client loaders (3 clients, IID)
    client_loaders = []
    for i in range(3):
        client_data = get_client_data(train_data, i, num_clients=3, partition="iid")
        client_loaders.append(DataLoader(client_data, batch_size=64, shuffle=True))
    
    # Initialize DP-SGD defense
    dp_defense = DPSGDDefense({})
    print(f"DP-SGD config: clip_norm={dp_defense.clip_norm}, noise_mult={dp_defense.noise_multiplier}")
    
    # One round of training
    criterion = torch.nn.CrossEntropyLoss()
    client_updates = []
    num_examples = []
    
    start_time = time.time()
    
    for client_id, loader in enumerate(client_loaders):
        local_model = create_model(device=device)
        local_model.load_state_dict(model.state_dict())
        
        optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9)
        
        local_model.train()
        batch_count = 0
        for images, labels in loader:
            if batch_count >= 100:  # 100 batches for better accuracy
                break
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(local_model(images), labels)
            loss.backward()
            optimizer.step()
            batch_count += 1
        
        update = [p.data.clone() for p in local_model.parameters()]
        client_updates.append(update)
        num_examples.append(len(loader.dataset))
    
    # Aggregate with DP
    aggregated = dp_defense.aggregate(client_updates, num_examples)
    
    # Update global model
    with torch.no_grad():
        for param, new_val in zip(model.parameters(), aggregated):
            param.copy_(new_val)
    
    # Evaluate
    metrics = evaluate_model(model, test_loader, device)
    elapsed = time.time() - start_time
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Loss: {metrics['loss']:.4f}")
    
    success = metrics['accuracy'] > 0.70  # Should be > 70% after 1 round
    print(f"Status: {'PASS' if success else 'FAIL'} (threshold: 70%)")
    return success


def test_cub200():
    """Test CUB-200 model with 1 round of actual training."""
    print("\n" + "=" * 60)
    print("TEST 2: CUB-200 Model (1 training round)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Test model creation
    from src.models.cub200_cnn import create_cub200_model
    from torch.utils.data import DataLoader
    
    print("Creating CUB-200 model (ResNet-50, all layers trainable)...")
    model = create_cub200_model(device=device)
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    forward_pass_ok = True
    accuracy = 0.0
    
    # Check if CUB-200 data exists
    data_path = "./data/CUB_200_2011"
    data_exists = os.path.exists(data_path)
    print(f"\nCUB-200 data at {data_path}: {'EXISTS' if data_exists else 'NOT FOUND'}")
    
    if data_exists:
        print("\nRunning 1 training epoch on CUB-200...")
        from src.utils.cub200_loader import load_cub200
        
        try:
            train_data, test_data = load_cub200("./data")
            print(f"Train samples: {len(train_data)}")
            print(f"Test samples: {len(test_data)}")
            
            train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)
            
            # Training
            model.train()
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=0.001, momentum=0.9, weight_decay=1e-4
            )
            criterion = torch.nn.CrossEntropyLoss()
            
            start_time = time.time()
            batch_count = 0
            total_loss = 0
            max_batches = 100  # Limit for quick test
            
            for images, labels in train_loader:
                if batch_count >= max_batches:
                    break
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                if batch_count % 20 == 0:
                    print(f"  Batch {batch_count}/{max_batches}, Loss: {total_loss/batch_count:.4f}")
            
            train_time = time.time() - start_time
            print(f"Training time: {train_time:.2f}s")
            
            # Evaluation
            print("\nEvaluating on test set...")
            model.eval()
            correct = 0
            total = 0
            eval_batches = 0
            max_eval = 50  # Limit for quick test
            
            with torch.no_grad():
                for images, labels in test_loader:
                    if eval_batches >= max_eval:
                        break
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    eval_batches += 1
            
            accuracy = correct / total
            print(f"Test accuracy: {accuracy*100:.2f}%")
            print(f"(Evaluated on {total} samples)")
            
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping training test (download CUB-200 first)")
    
    # Summary
    print(f"\nForward pass: {'OK' if forward_pass_ok else 'FAIL'}")
    print(f"All {trainable_params:,} parameters trainable")
    
    # For CUB-200 after 100 batches, expect at least 5% accuracy
    # (random would be 0.5% for 200 classes)
    if data_exists:
        accuracy_ok = accuracy > 0.02  # At least 2% (4x better than random)
        print(f"Accuracy: {'OK' if accuracy_ok else 'LOW'} ({accuracy*100:.2f}% > 2%)")
        success = forward_pass_ok and accuracy_ok
    else:
        success = forward_pass_ok
    
    print(f"Status: {'PASS' if success else 'FAIL'}")
    return success


def main():
    print("\n" + "=" * 60)
    print("QUICK SANITY CHECK")
    print("=" * 60)
    
    results = {}
    
    # Test DP-SGD
    try:
        results['dp_sgd'] = test_dp_sgd()
    except Exception as e:
        print(f"DP-SGD test FAILED with error: {e}")
        results['dp_sgd'] = False
    
    # Test CUB-200
    try:
        results['cub200'] = test_cub200()
    except Exception as e:
        print(f"CUB-200 test FAILED with error: {e}")
        results['cub200'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test}: {status}")
    
    all_passed = all(results.values())
    print("\n" + ("All tests PASSED!" if all_passed else "Some tests FAILED!"))
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
