def coldstart_predictions_data():
    list_true = []
    list_predict = []
    
    # Group cold-start orders
    cold_groups = cold_start.groupby('order_id')
    
    for order_id, group in tqdm(cold_groups):
        products = group['product_id'].tolist()
        
        # Skip orders with less than 5 items
        if len(products) < 5:
            continue
        
        # Encode product IDs using existing mapping
        encoded_products = [num_mapping[p] for p in products if p in num_mapping]
        
        # Use first 4 products as input
        input_seq = encoded_products[:4]
        
        # Ground truth = remaining items
        true_next = encoded_products[4:]
        
        # Generate predictions
        context = torch.tensor(input_seq, device=device).unsqueeze(0)
        predictions = m.generate(context, max_new_tokens=10)[0].tolist()
        predictions = predictions[-10:]  # Keep last 10 tokens
        
        list_true.append(true_next)
        list_predict.append(predictions)
    
    return list_true, list_predict