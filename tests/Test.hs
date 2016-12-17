module Main (
	main
	) where

import Numeric.Trainee.Neural

test ∷ IO (Net Double)
test = net $ input 10 ⭃ fc sigma 10 ⭃ fc sigma 5 ⭃ fc sigma 1

main ∷ IO ()
main = putStrLn "Hello"
