function confusionMatrix(predicted, target, Classes)
	local confMatrix = torch.Tensor(#Classes, #Classes):fill(0)
	local subconfMat = torch.Tensor(#Classes):fill(0)
	for targetIDX = 1, #Classes do
		local matches = predicted:maskedSelect(target:eq(targetIDX))
		for predIDX = 1, #Classes do
			subconfMat[predIDX] = torch.sum(matches:eq(predIDX))
		end
		confMatrix[{ {targetIDX},{} }] = subconfMat
	end	

	local acc = torch.Tensor(#Classes):fill(0)

	for targetIDX = 1, #Classes do
		acc[targetIDX] = confMatrix[targetIDX][targetIDX]/torch.sum(confMatrix[{ {targetIDX},{} }])*100		
	end

	for col=1,#Classes do
		local assign = '[' .. "\t"
		for row=1,#Classes do
			if row == #Classes  then
				assign = assign .. confMatrix[row][col] .. "\t]"
			else
				assign = assign .. confMatrix[row][col] .. "\t"
			end
		end
		print(string.format(assign .. " %2.2f\t\t{%s}",acc[col],Classes[col]))
--		assign = assign .. "  " .. acc[col] .. "  "
--		assign = assign .. "\t" .. "{" .. Classes[col] .. "}"
--`	print(assign)
	end	

	return confMatrix, acc

end
