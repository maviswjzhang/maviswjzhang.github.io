```python
    def pearsonDistance(self, x, y):
        # b*sqrt(k)*sqrt(k), c, S^2  +  b*sqrt(k)*sqrt(k), c, 3^2  -->  b*sqrt(k)*sqrt(k), S^2, 3^2
        N = x.shape[1]

        sumx = torch.sum(x, dim=1)  # b*sqrt(k)*sqrt(k), S^2
        sumy = torch.sum(y, dim=1)  # b*sqrt(k)*sqrt(k), 3^2

        sqx = torch.pow(x, 2)
        sumSqx = torch.sum(sqx, dim=1)  # b*sqrt(k)*sqrt(k), S^2
        sqy = torch.pow(y, 2)
        sumSqy = torch.sum(sqy, dim=1)  # b*sqrt(k)*sqrt(k), 3^2

        xt = torch.transpose(x, 2, 1)
        xty = torch.bmm(xt, y)  # b*sqrt(k)*sqrt(k), S^2, 3^2

        numerator = xty - torch.bmm(torch.unsqueeze(sumx, dim=2), torch.unsqueeze(sumy, dim=1)) / N

        denx = sumSqx - torch.pow(sumx, 2) / N  # b*sqrt(k)*sqrt(k), S^2
        deny = sumSqy - torch.pow(sumy, 2) / N  # b*sqrt(k)*sqrt(k), 3^2
        denominator = torch.sqrt(torch.bmm(torch.unsqueeze(denx, dim=2), torch.unsqueeze(deny, dim=1)))

        return numerator / denominator

    def cosineDistance(self, x, y):
        # b*sqrt(k)*sqrt(k), c, S^2  +  b*sqrt(k)*sqrt(k), c, 3^2  -->  b*sqrt(k)*sqrt(k), S^2, 3^2
        N = x.shape[1]

        xt = torch.transpose(x, 2, 1)
        xty = torch.bmm(xt, y)  # b*sqrt(k)*sqrt(k), S^2, 3^2

        sqx = torch.pow(x, 2)
        sumSqx = torch.sqrt(torch.sum(sqx, dim=1))  # b*sqrt(k)*sqrt(k), S^2
        sumSqx = torch.unsqueeze(sumSqx, dim=2)
        sqy = torch.pow(y, 2)
        sumSqy = torch.sqrt(torch.sum(sqy, dim=1))  # b*sqrt(k)*sqrt(k), 3^2
        sumSqy = torch.unsqueeze(sumSqy, dim=1)

        cosine_distance = xty / (sumSqx * sumSqy)

        return cosine_distance
```

